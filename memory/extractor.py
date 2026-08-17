from __future__ import annotations

import re
import time

from memory.models import (
    ExtractionResult,
    MemoryCandidate,
    MemoryScope,
    MemorySource,
    MemoryType,
)


class MemoryExtractor:
    """Conservative rule-based V1 extraction into the same typed candidate contract an LLM can use later."""

    def extract(self, text: str, *, project_id: str | None) -> ExtractionResult:
        started = time.perf_counter()
        clean_text = " ".join(text.strip().split())
        explicit_text = re.sub(r"^remember(?: that)?\s+", "", clean_text, flags=re.IGNORECASE)
        candidates: list[MemoryCandidate] = []

        candidates.extend(self._profile_candidates(explicit_text))
        candidates.extend(self._preference_candidates(explicit_text))
        candidates.extend(self._project_candidates(explicit_text, project_id))

        is_explicit_command = explicit_text != clean_text
        if is_explicit_command and not candidates and explicit_text:
            candidates.append(
                MemoryCandidate(
                    memory_type=MemoryType.SEMANTIC,
                    scope=MemoryScope.USER,
                    content=explicit_text,
                    source=MemorySource.USER_EXPLICIT,
                    confidence=0.98,
                    importance=0.7,
                )
            )

        unique: dict[tuple[str, str], MemoryCandidate] = {}
        for candidate in candidates:
            unique[(candidate.key, candidate.content.lower())] = candidate
        return ExtractionResult(
            candidates=tuple(unique.values()),
            elapsed_seconds=time.perf_counter() - started,
        )

    @staticmethod
    def _profile_candidates(text: str) -> list[MemoryCandidate]:
        patterns = (
            (
                r"\bmy name is\s+(?P<value>[^.!?]{1,100})",
                "profile.name",
                lambda value: f"User's name is {value}",
                0.95,
            ),
            (
                r"\bmy (?:long[- ]term )?goal is\s+(?P<value>[^.!?]{1,300})",
                "profile.long_term_goal",
                lambda value: f"User's long-term goal is {value}",
                0.95,
            ),
            (
                r"\bi want to become\s+(?P<value>[^.!?]{1,200})",
                "profile.long_term_goal",
                lambda value: f"User wants to become {value}",
                0.9,
            ),
        )
        results: list[MemoryCandidate] = []
        for pattern, key, formatter, importance in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group("value").strip(" ,")
                results.append(
                    MemoryCandidate(
                        memory_type=MemoryType.PROFILE,
                        scope=MemoryScope.USER,
                        key=key,
                        content=formatter(value),
                        source=MemorySource.USER_EXPLICIT,
                        confidence=0.98,
                        importance=importance,
                    )
                )
        return results

    @staticmethod
    def _preference_candidates(text: str) -> list[MemoryCandidate]:
        results: list[MemoryCandidate] = []
        favorite = re.search(
            r"\bmy favou?rite programming language is\s+(?P<value>[^.!?]{1,100})",
            text,
            re.IGNORECASE,
        )
        if favorite:
            language = favorite.group("value").strip(" ,")
            results.append(
                MemoryCandidate(
                    memory_type=MemoryType.PROFILE,
                    scope=MemoryScope.USER,
                    key="preference.programming_language",
                    content=f"User's favorite programming language is {language}",
                    source=MemorySource.USER_EXPLICIT,
                    confidence=0.99,
                    importance=0.8,
                )
            )

        backend_patterns = (
            r"\bi (?:currently )?use\s+(?P<value>[A-Za-z0-9+#. -]{1,50})\s+for (?:my )?backend",
            r"\bi(?:'m| am) moving (?:my )?backend (?:projects? )?to\s+(?P<value>[A-Za-z0-9+#. -]{1,50})",
            r"\bmy current backend language is\s+(?P<value>[A-Za-z0-9+#. -]{1,50})",
        )
        for pattern in backend_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                language = match.group("value").strip(" ,")
                results.append(
                    MemoryCandidate(
                        memory_type=MemoryType.PROFILE,
                        scope=MemoryScope.USER,
                        key="preference.backend_language",
                        content=f"User's current backend language is {language}",
                        source=MemorySource.USER_EXPLICIT,
                        confidence=0.98,
                        importance=0.8,
                    )
                )
                break

        preference = re.search(r"\bi prefer\s+(?P<value>[^.!?]{1,300})", text, re.IGNORECASE)
        if preference and not favorite:
            value = preference.group("value").strip(" ,")
            lower = value.lower()
            if "explanation" in lower or "diagram" in lower:
                key = "preference.explanation_style"
            elif "backend" in lower:
                key = "preference.backend_language"
            elif any(word in lower for word in ("python", "javascript", "typescript", " go ", "language")):
                key = "preference.programming_language"
            else:
                first_word = re.findall(r"[a-z0-9]+", lower)
                key = f"preference.{first_word[0] if first_word else 'general'}"
            results.append(
                MemoryCandidate(
                    memory_type=MemoryType.SEMANTIC,
                    scope=MemoryScope.USER,
                    key=key,
                    content=f"User prefers {value}",
                    source=MemorySource.USER_EXPLICIT,
                    confidence=0.98,
                    importance=0.75,
                )
            )
        return results

    @staticmethod
    def _project_candidates(text: str, project_id: str | None) -> list[MemoryCandidate]:
        if not project_id:
            return []
        build = re.search(
            r"\b(?:i am|i'm|we are|we're) building\s+(?P<value>[^.!?]{1,300})",
            text,
            re.IGNORECASE,
        )
        if not build:
            return []
        value = build.group("value").strip(" ,")
        return [
            MemoryCandidate(
                memory_type=MemoryType.PROJECT,
                scope=MemoryScope.PROJECT,
                project_id=project_id,
                key="project.current_build",
                content=f"The current project is building {value}",
                source=MemorySource.USER_EXPLICIT,
                confidence=0.96,
                importance=0.8,
            )
        ]
