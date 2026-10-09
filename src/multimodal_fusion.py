"""
Multimodal Emotion Fusion Module.
Combines visual facial expressions and acoustic speech emotion predictions
into an integrated affective computing decision with congruency analysis.
"""

from typing import Dict, Any, Optional
import numpy as np
from src.config import EMOTIONS, normalize_emotion


class MultimodalEmotionFusion:
    """Fuses facial and audio emotion recognition predictions."""

    @staticmethod
    def fuse(
        face_result: Dict[str, Any],
        audio_result: Dict[str, Any],
        face_weight: float = 0.5,
        audio_weight: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Combine facial and audio predictions using weighted decision-level fusion.
        
        Args:
            face_result: Output dictionary from FacialEmotionDetector.predict()
            audio_result: Output dictionary from AudioEmotionDetector.predict()
            face_weight: Relative weight for facial expression (0.0 to 1.0)
            audio_weight: Relative weight for acoustic speech (0.0 to 1.0)
            
        Returns:
            Dictionary containing fused emotion, confidence, congruency, and breakdown.
        """
        # Ensure weights are positive and normalized
        total_weight = face_weight + audio_weight
        if total_weight <= 0:
            face_weight, audio_weight = 0.5, 0.5
            total_weight = 1.0
        w_f = face_weight / total_weight
        w_a = audio_weight / total_weight

        # Extract probability distributions
        face_probs = face_result.get("primary_probabilities") or face_result.get("probabilities", {})
        audio_probs = audio_result.get("probabilities", {})

        # Compute fused probabilities for each canonical emotion
        combined_probs = {}
        for emotion in EMOTIONS:
            p_f = float(face_probs.get(emotion, 0.0))
            p_a = float(audio_probs.get(emotion, 0.0))
            combined_probs[emotion] = round(w_f * p_f + w_a * p_a, 4)

        # Re-normalize combined distribution
        prob_sum = sum(combined_probs.values())
        if prob_sum > 0:
            for emotion in combined_probs:
                combined_probs[emotion] = round(combined_probs[emotion] / prob_sum, 4)

        # Primary predictions from each modality
        face_top = normalize_emotion(face_result.get("primary_emotion") or face_result.get("emotion", "Neutral"))
        face_conf = float(face_result.get("primary_confidence") or face_result.get("confidence", 0.0))

        audio_top = normalize_emotion(audio_result.get("emotion", "Neutral"))
        audio_conf = float(audio_result.get("confidence", 0.0))

        # Top fused emotion
        fused_emotion = max(combined_probs, key=combined_probs.get)
        fused_conf = combined_probs[fused_emotion]

        # Congruency Analysis
        is_congruent = (face_top == audio_top)

        # Vector cosine similarity between face and audio probability distributions
        v_face = np.array([face_probs.get(e, 0.0) for e in EMOTIONS], dtype=np.float32)
        v_audio = np.array([audio_probs.get(e, 0.0) for e in EMOTIONS], dtype=np.float32)
        norm_f = np.linalg.norm(v_face)
        norm_a = np.linalg.norm(v_audio)
        if norm_f > 0 and norm_a > 0:
            similarity = float(np.dot(v_face, v_audio) / (norm_f * norm_a))
        else:
            similarity = 1.0 if is_congruent else 0.0

        insight = MultimodalEmotionFusion._generate_affective_insight(
            face_top=face_top,
            face_conf=face_conf,
            audio_top=audio_top,
            audio_conf=audio_conf,
            fused_emotion=fused_emotion,
            fused_conf=fused_conf,
            is_congruent=is_congruent,
        )

        return {
            "success": True,
            "fused_emotion": fused_emotion,
            "fused_confidence": fused_conf,
            "is_congruent": is_congruent,
            "congruency_score": round(similarity, 3),
            "face": {
                "emotion": face_top,
                "confidence": round(face_conf, 4),
                "weight": round(w_f, 2),
                "probabilities": face_probs,
            },
            "audio": {
                "emotion": audio_top,
                "confidence": round(audio_conf, 4),
                "weight": round(w_a, 2),
                "probabilities": audio_probs,
            },
            "combined_probabilities": combined_probs,
            "insight": insight,
        }

    @staticmethod
    def _generate_affective_insight(
        face_top: str,
        face_conf: float,
        audio_top: str,
        audio_conf: float,
        fused_emotion: str,
        fused_conf: float,
        is_congruent: bool,
    ) -> str:
        """Generate human-readable affective computing insight."""
        if is_congruent:
            return (
                f"High-Confidence Emotional Alignment ({fused_conf * 100:.1f}%): "
                f"Both visual facial expression ({face_top} at {face_conf * 100:.1f}%) and acoustic vocal cues "
                f"({audio_top} at {audio_conf * 100:.1f}%) synchronously convey '{fused_emotion}'."
            )

        # Incongruent cases provide rich affective interpretation
        pair = (face_top, audio_top)
        if pair == ("Happy", "Angry") or pair == ("Happy", "Disgust"):
            return (
                f"Emotional Dissonance: Facial smile ({face_top} {face_conf*100:.1f}%) combined with "
                f"hostile vocal tone ({audio_top} {audio_conf*100:.1f}%). Suggests passive aggression, "
                f"forced social masking, or intense sarcasm. Fused result leans toward '{fused_emotion}'."
            )
        elif pair == ("Neutral", "Sad") or pair == ("Happy", "Sad"):
            return (
                f"Emotional Masking Detected: Face presents {face_top} ({face_conf*100:.1f}%) but vocal acoustics "
                f"reveal underlying sadness ({audio_top} {audio_conf*100:.1f}%). Often observed in concealed distress."
            )
        elif pair == ("Angry", "Fear") or pair == ("Fear", "Angry"):
            return (
                f"Fight-or-Flight Acute Arousal: Conflicting fear and anger cues detected between face ({face_top}) "
                f"and voice ({audio_top}). Indicates high-stress emotional reaction. Integrated state: '{fused_emotion}'."
            )
        else:
            return (
                f"Multimodal Discrepancy: Visual cues signal {face_top} ({face_conf*100:.1f}%) while speech prosody "
                f"indicates {audio_top} ({audio_conf*100:.1f}%). Fused weighted consensus: '{fused_emotion}' ({fused_conf*100:.1f}%)."
            )
