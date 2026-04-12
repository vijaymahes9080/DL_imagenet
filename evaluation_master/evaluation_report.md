# 🔬 Final Model Evaluation Report

**Model**: models/autonomous/ensemble_backbone.keras
**Target Modality**: Emotion (Projected)
**Accuracy**: 96.51%
**Verdict**: DEPLOY

## Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

## Metrics
```
              precision    recall  f1-score   support

       Angry       0.95      0.97      0.96       452
     Disgust       0.72      1.00      0.84        38
        Fear       1.00      0.96      0.98       478
       Happy       0.98      0.96      0.97       845
     Neutral       0.95      0.96      0.95       573
         Sad       0.92      0.98      0.95       286
    Surprise       0.99      0.96      0.98       596

    accuracy                           0.97      3268
   macro avg       0.93      0.97      0.95      3268
weighted avg       0.97      0.97      0.97      3268

```
