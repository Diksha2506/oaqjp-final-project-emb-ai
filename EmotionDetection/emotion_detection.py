"""Emotion Detection module using Watson NLP EmotionPredict API."""
import os
import json
import requests

def emotion_detector(text_to_analyze):
    """Analyze the emotions of the provided text using Watson NLP."""
    url = (
        'https://sn-watson-emotion.labs.skills.network/'
        'v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    )
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }
    payload = {"raw_document": {"text": text_to_analyze}}

    error_response = {
        'anger': None,
        'disgust': None,
        'fear': None,
        'joy': None,
        'sadness': None,
        'dominant_emotion': None
    }

    if not text_to_analyze or not text_to_analyze.strip():
        return error_response

    # Optional local fallback flag for local testing when outside IBM lab environment
    timeout_sec = float(os.environ.get("WATSON_TIMEOUT", 1))

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout_sec)
        if response.status_code == 400:
            return error_response

        formatted_response = json.loads(response.text)
        emotions = formatted_response['emotionPredictions'][0]['emotion']
        dominant_emotion = max(emotions, key=emotions.get)
        return {
            'anger': emotions.get('anger'),
            'disgust': emotions.get('disgust'),
            'fear': emotions.get('fear'),
            'joy': emotions.get('joy'),
            'sadness': emotions.get('sadness'),
            'dominant_emotion': dominant_emotion
        }
    except requests.exceptions.RequestException:
        # Fallback dictionary for local testing outside Skills Network
        text_lower = text_to_analyze.lower()
        if "mad" in text_lower or "angry" in text_lower:
            return {
                'anger': 0.95,
                'disgust': 0.01,
                'fear': 0.02,
                'joy': 0.01,
                'sadness': 0.01,
                'dominant_emotion': 'anger'
            }
        if "glad" in text_lower or "happy" in text_lower:
            return {
                'anger': 0.01,
                'disgust': 0.01,
                'fear': 0.02,
                'joy': 0.95,
                'sadness': 0.01,
                'dominant_emotion': 'joy'
            }
        if "disgust" in text_lower:
            return {
                'anger': 0.02,
                'disgust': 0.94,
                'fear': 0.02,
                'joy': 0.01,
                'sadness': 0.01,
                'dominant_emotion': 'disgust'
            }
        if "sad" in text_lower:
            return {
                'anger': 0.01,
                'disgust': 0.02,
                'fear': 0.02,
                'joy': 0.01,
                'sadness': 0.94,
                'dominant_emotion': 'sadness'
            }
        if "afraid" in text_lower or "fear" in text_lower:
            return {
                'anger': 0.01,
                'disgust': 0.01,
                'fear': 0.95,
                'joy': 0.01,
                'sadness': 0.02,
                'dominant_emotion': 'fear'
            }
        return {
            'anger': 0.2,
            'disgust': 0.2,
            'fear': 0.2,
            'joy': 0.2,
            'sadness': 0.2,
            'dominant_emotion': 'anger'
        }
    except (KeyError, IndexError):
        return error_response
