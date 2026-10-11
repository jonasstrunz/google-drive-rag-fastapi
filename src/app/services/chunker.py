import re
from typing import List

import tiktoken


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        model_name: str = "gpt-4o-mini",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def split_text(self, text: str) -> List[str]:
        """Splits text into semantically cohesive overlapping chunks based on tokens."""
        if not text or not text.strip():
            return []

        # Split into paragraph / sentence blocks
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        chunks: List[str] = []
        current_chunk_tokens: List[int] = []
        current_chunk_text: List[str] = []

        for paragraph in paragraphs:
            para_tokens = self.encoding.encode(paragraph)

            # If a single paragraph is larger than chunk_size, split by sentences or sliding token window
            if len(para_tokens) > self.chunk_size:
                # Flush existing buffer
                if current_chunk_text:
                    chunks.append("\n\n".join(current_chunk_text).strip())
                    current_chunk_text = []
                    current_chunk_tokens = []

                # Split long paragraph using sliding token window
                start = 0
                while start < len(para_tokens):
                    end = min(start + self.chunk_size, len(para_tokens))
                    chunk_slice = para_tokens[start:end]
                    chunk_str = self.encoding.decode(chunk_slice)
                    if chunk_str.strip():
                        chunks.append(chunk_str.strip())
                    if end == len(para_tokens):
                        break
                    start += max(1, self.chunk_size - self.chunk_overlap)
                continue

            # Check if adding this paragraph exceeds chunk size
            if len(current_chunk_tokens) + len(para_tokens) > self.chunk_size:
                if current_chunk_text:
                    chunks.append("\n\n".join(current_chunk_text).strip())

                # Overlap: keep recent tokens or paragraphs if possible
                overlap_text = []
                overlap_token_count = 0
                for prev_para in reversed(current_chunk_text):
                    prev_tokens_len = len(self.encoding.encode(prev_para))
                    if overlap_token_count + prev_tokens_len <= self.chunk_overlap:
                        overlap_text.insert(0, prev_para)
                        overlap_token_count += prev_tokens_len
                    else:
                        break

                current_chunk_text = overlap_text + [paragraph]
                current_chunk_tokens = self.encoding.encode("\n\n".join(current_chunk_text))
            else:
                current_chunk_text.append(paragraph)
                current_chunk_tokens = self.encoding.encode("\n\n".join(current_chunk_text))

        if current_chunk_text:
            remaining = "\n\n".join(current_chunk_text).strip()
            if remaining:
                chunks.append(remaining)

        return chunks
