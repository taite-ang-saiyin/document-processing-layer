from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

from app.config import settings
from app.core.alignment import ImageAligner
from app.models.schemas import TemplateDefinition
from app.services.template_references import TemplateReferenceStore, template_reference_store


@dataclass(frozen=True)
class TemplateMatchCandidate:
    template_id: str
    score: float


@dataclass(frozen=True)
class TemplateMatchResult:
    selected_template_id: Optional[str]
    candidates: List[TemplateMatchCandidate]
    reason: Optional[str] = None

    @property
    def top_score(self) -> Optional[float]:
        return self.candidates[0].score if self.candidates else None

    @property
    def runner_up_score(self) -> Optional[float]:
        return self.candidates[1].score if len(self.candidates) > 1 else None

    def as_error_detail(self) -> dict:
        return {
            "code": "TEMPLATE_MATCH_UNCERTAIN",
            "message": self.reason or "No template could be selected with confidence.",
            "candidates": [asdict(candidate) for candidate in self.candidates],
        }


class TemplateMatcher:
    """Selects a template from its canonical reference image using ORB alignment scores."""

    def __init__(
        self,
        reference_store: TemplateReferenceStore = template_reference_store,
        score_threshold: float = settings.TEMPLATE_MATCH_SCORE_THRESHOLD,
        score_margin: float = settings.TEMPLATE_MATCH_MARGIN,
    ):
        self.reference_store = reference_store
        self.score_threshold = score_threshold
        self.score_margin = score_margin
        self.aligner = ImageAligner()

    def match(
        self,
        image_np: np.ndarray,
        templates: Dict[str, TemplateDefinition],
    ) -> TemplateMatchResult:
        candidates: List[TemplateMatchCandidate] = []
        for template_id in templates:
            reference_np = self.reference_store.load(template_id)
            if reference_np is None:
                continue
            _, _, score = self.aligner.align_images(image_np, reference_np)
            candidates.append(TemplateMatchCandidate(template_id=template_id, score=score))

        candidates.sort(key=lambda candidate: (-candidate.score, candidate.template_id))
        if not candidates:
            return TemplateMatchResult(
                selected_template_id=None,
                candidates=[],
                reason="No registered templates have a usable reference image.",
            )

        top = candidates[0]
        if top.score < self.score_threshold:
            return TemplateMatchResult(
                selected_template_id=None,
                candidates=candidates,
                reason=(
                    f"Best template score {top.score:.3f} is below the required "
                    f"threshold {self.score_threshold:.3f}."
                ),
            )

        if len(candidates) > 1:
            runner_up = candidates[1]
            score_gap = top.score - runner_up.score
            if score_gap < self.score_margin:
                return TemplateMatchResult(
                    selected_template_id=None,
                    candidates=candidates,
                    reason=(
                        f"Best template score leads the runner-up by {score_gap:.3f}, below the "
                        f"required margin {self.score_margin:.3f}."
                    ),
                )

        return TemplateMatchResult(selected_template_id=top.template_id, candidates=candidates)
