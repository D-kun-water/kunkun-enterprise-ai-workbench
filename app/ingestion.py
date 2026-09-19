"""Document loading, heading-aware chunking, and ingestion persistence."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

from app.utils import ensure_directory, normalize_text


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf", ".docx"}
WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def is_catalog_text(text: str) -> bool:
    """Identify a standalone TOC, not a body page that merely mentions one."""
    lines = [re.sub(r"\s+", "", line) for line in text.splitlines() if line.strip()]
    if not any(line == "目录" for line in lines[:3]):
        return False
    # Mixed TOC/body pages keep their prose. A standalone TOC consists of short
    # headings, page numbers and running headers, with no substantive sentences.
    return sum(bool(re.match(r"^(?:第[一二三四五六七八九十百\d]+章|\d+$)", line)) for line in lines) >= 4 and not any(
        len(line) > 45 or "。" in line or "；" in line for line in lines
    )


@dataclass
class KnowledgeDocument:
    document_id: str
    title: str
    category: str
    source_path: str
    content: str
    sections: list[str]
    status: str = "ready"
    error: str = ""
    version: str = ""
    effective_date: str = ""
    department: str = ""
    owner: str = ""
    knowledge_base_id: str = "policy"
    allowed_roles: list[str] = field(default_factory=list)


@dataclass
class TextChunk:
    chunk_id: str
    document_id: str
    title: str
    category: str
    section: str
    source_path: str
    content: str
    chunk_index: int
    knowledge_base_id: str = "policy"
    allowed_roles: list[str] = field(default_factory=list)


class DocumentLoader:
    """Load the file types supported by the local knowledge base."""

    def load(self, path: Path) -> KnowledgeDocument:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文档格式：{suffix or '无扩展名'}")
        if not path.exists():
            raise FileNotFoundError(path)

        if suffix == ".pdf":
            content = self._load_pdf(path)
            title = self._load_pdf_title(path)
        elif suffix == ".docx":
            content = self._load_docx(path)
            title = self._extract_title(content)
        else:
            content = path.read_text(encoding="utf-8-sig")
            title = self._extract_title(content) if suffix in {".md", ".markdown"} else ""

        content = normalize_text(content)
        if not content:
            raise ValueError(f"文档内容为空：{path.name}")

        title = title or self._extract_title(content)
        sections = self._extract_sections(content)
        policy_metadata = self._extract_policy_metadata(content)
        return KnowledgeDocument(
            document_id=path.stem,
            title=title or path.stem,
            category=self._extract_category(content, path.name),
            source_path=path.name,
            content=content,
            sections=sections,
            **policy_metadata,
        )

    def load_directory(self, directory: Path) -> list[KnowledgeDocument]:
        directory = Path(directory)
        if not directory.exists():
            return []
        paths = sorted(
            path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        documents: list[KnowledgeDocument] = []
        for path in paths:
            try:
                documents.append(self.load(path))
            except Exception as exc:  # A bad file should remain visible in status output.
                documents.append(
                    KnowledgeDocument(
                        document_id=path.stem,
                        title=path.stem,
                        category=self._infer_category(path.name),
                        source_path=path.name,
                        content="",
                        sections=[],
                        status="error",
                        error=str(exc),
                    )
                )
        used_ids: set[str] = set()
        for document in documents:
            document_id = document.document_id
            if document_id in used_ids:
                suffix = Path(document.source_path).suffix.lower().lstrip(".") or "file"
                document_id = f"{document_id}-{suffix}"
                counter = 2
                while document_id in used_ids:
                    document_id = f"{document.document_id}-{suffix}-{counter}"
                    counter += 1
                document.document_id = document_id
            used_ids.add(document.document_id)
        return documents

    @staticmethod
    def _load_pdf(path: Path) -> str:
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - exercised only without PyMuPDF
            raise RuntimeError("读取 PDF 需要安装 PyMuPDF") from exc

        pages: list[str] = []
        with fitz.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text("text").strip()
                if page_text:
                    pages.append(f"## 第 {page_number} 页\n{page_text}")
        return "\n\n".join(pages)

    @staticmethod
    def _load_pdf_title(path: Path) -> str:
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - exercised only without PyMuPDF
            raise RuntimeError("读取 PDF 需要安装 PyMuPDF") from exc

        with fitz.open(path) as pdf:
            return str(pdf.metadata.get("title") or "").strip()

    @staticmethod
    def _load_docx(path: Path) -> str:
        try:
            with zipfile.ZipFile(path) as archive:
                root = ElementTree.fromstring(archive.read("word/document.xml"))
        except (KeyError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
            raise ValueError(f"无法读取 Word 文档：{path.name}") from exc

        lines: list[str] = []
        title_found = False
        for paragraph in root.iter(f"{WORD_NAMESPACE}p"):
            text = "".join(node.text or "" for node in paragraph.iter(f"{WORD_NAMESPACE}t")).strip()
            if not text:
                continue
            style = paragraph.find(f"{WORD_NAMESPACE}pPr/{WORD_NAMESPACE}pStyle")
            style_name = style.get(f"{WORD_NAMESPACE}val", "").lower() if style is not None else ""
            level_match = re.fullmatch(r"heading(\d+)", style_name)
            if style_name == "title" and not title_found:
                lines.append(f"# {text}")
                title_found = True
            elif level_match:
                level = min(int(level_match.group(1)) + 1, 4)
                lines.append(f"{'#' * level} {text}")
            else:
                lines.append(text)
        return "\n\n".join(lines)

    @staticmethod
    def _extract_title(content: str) -> str:
        metadata_match = re.search(r"文档名称\s*[：:]\s*([^\n]+)", content)
        if metadata_match:
            return metadata_match.group(1).strip(" -*")
        match = re.search(r"^#\s+(.+?)\s*$", content, flags=re.MULTILINE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_sections(content: str) -> list[str]:
        return [
            match.group(1).strip()
            for match in re.finditer(r"^#{2,4}\s+(.+?)\s*$", content, flags=re.MULTILINE)
        ]

    def _extract_category(self, content: str, filename: str) -> str:
        match = re.search(
            r"(?:知识分类|业务分类|所属分类|分类)\s*[：:]\s*([^\n]+)",
            content,
            flags=re.IGNORECASE,
        )
        return match.group(1).strip(" -*") if match else self._infer_category(filename)

    @staticmethod
    def _infer_category(filename: str) -> str:
        lowered = filename.lower()
        category_hints = {
            "人力资源": ("onboarding", "probation", "attendance", "leave", "offboarding", "hr"),
            "财务采购": ("travel", "reimbursement", "invoice", "expense", "procurement", "finance"),
            "IT 服务": ("account", "password", "vpn", "software", "it_"),
            "信息安全": ("security", "information_security"),
            "行政服务": ("equipment", "meeting", "access", "admin"),
        }
        for category, hints in category_hints.items():
            if any(hint in lowered for hint in hints):
                return category
        return "企业制度"

    @staticmethod
    def _extract_policy_metadata(content: str) -> dict[str, str]:
        patterns = {
            "version": r"版本\s*[：:]\s*([^\n]+)",
            "effective_date": r"生效日期\s*[：:]\s*([^\n]+)",
            "department": r"(?:发布部门|责任部门|归口部门)\s*[：:]\s*([^\n]+)",
            # Match a standalone metadata label.  A broader substring match
            # would treat lines such as "部门负责人：负责预算确认" as a
            # person's name.
            "owner": r"(?:^|\n)\s*(?:[-*]\s*)?负责人\s*[：:]\s*([^\n]+)",
        }
        values: dict[str, str] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, content, flags=re.IGNORECASE)
            values[key] = match.group(1).strip(" -*") if match else ""
        return values


class TextChunker:
    """Split documents by headings first, then by overlapping character windows."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须大于等于 0 且小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_document(self, document: KnowledgeDocument) -> list[TextChunk]:
        if document.status != "ready" or not document.content:
            return []

        sections = self._section_blocks(document.content)
        chunks: list[TextChunk] = []
        for section_name, section_text in sections:
            section_chunks = self.split_text(
                section_text,
                document_id=document.document_id,
                title=document.title,
                category=document.category,
                section=section_name,
                source_path=document.source_path,
                start_index=len(chunks),
                knowledge_base_id=document.knowledge_base_id,
                allowed_roles=document.allowed_roles,
            )
            chunks.extend(section_chunks)
        return chunks

    def split_documents(self, documents: Iterable[KnowledgeDocument]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for document in documents:
            chunks.extend(self.split_document(document))
        return chunks

    def split_text(
        self,
        text: str,
        *,
        document_id: str,
        title: str,
        category: str = "",
        section: str = "正文",
        source_path: str = "",
        start_index: int = 0,
        knowledge_base_id: str = "policy",
        allowed_roles: list[str] | None = None,
    ) -> list[TextChunk]:
        text = normalize_text(text)
        if not text:
            return []

        chunks: list[TextChunk] = []
        step = self.chunk_size - self.chunk_overlap
        offset = 0
        chunk_index = start_index
        while offset < len(text):
            content = text[offset : offset + self.chunk_size]
            chunks.append(
                TextChunk(
                    chunk_id=f"{document_id}-{chunk_index:03d}",
                    document_id=document_id,
                    title=title,
                    category=category,
                    section=section or "正文",
                    source_path=source_path,
                    content=content,
                    chunk_index=chunk_index,
                    knowledge_base_id=knowledge_base_id,
                    allowed_roles=list(allowed_roles or []),
                )
            )
            if offset + self.chunk_size >= len(text):
                break
            offset += step
            chunk_index += 1
        return chunks

    @staticmethod
    def _section_blocks(content: str) -> list[tuple[str, str]]:
        heading_pattern = re.compile(r"^##\s+(.+?)\s*$", flags=re.MULTILINE)
        matches = list(heading_pattern.finditer(content))
        if not matches:
            return [("正文", content)]

        blocks: list[tuple[str, str]] = []
        # The preamble contains the disclaimer and catalog metadata. It remains
        # on KnowledgeDocument for status display, but excluding it from search
        # chunks keeps employee answers focused on actual policy sections.
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            body = content[start:end].strip()
            section_name = match.group(1).strip()
            if body and section_name not in {"文档信息", "文档元数据", "基本信息"} and not is_catalog_text(body):
                blocks.append((section_name, body))
        return blocks


class IngestionStore:
    """Persist parsed artifacts as inspectable JSON files."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def save(
        self,
        documents: list[KnowledgeDocument],
        chunks: list[TextChunk],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        ensure_directory(self.directory)
        self._write_json("documents.json", [asdict(document) for document in documents])
        self._write_json("chunks.json", [asdict(chunk) for chunk in chunks])
        self._write_json("index_meta.json", metadata or {})

    def load(self) -> tuple[list[KnowledgeDocument], list[TextChunk], dict[str, Any]]:
        documents = [KnowledgeDocument(**item) for item in self._read_json("documents.json")]
        chunks = [TextChunk(**item) for item in self._read_json("chunks.json")]
        metadata = self._read_json("index_meta.json")
        return documents, chunks, metadata

    def exists(self) -> bool:
        required = ("documents.json", "chunks.json", "index_meta.json")
        return all((self.directory / filename).exists() for filename in required)

    def _write_json(self, filename: str, value: Any) -> None:
        path = self.directory / filename
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_json(self, filename: str) -> Any:
        path = self.directory / filename
        return json.loads(path.read_text(encoding="utf-8"))
