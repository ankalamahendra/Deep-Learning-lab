import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Example true (actual) and predicted labels
actual    = np.array(['Positive', 'Negative', 'Positive', 'Positive', 'Negative', 
                      'Negative', 'Positive', 'Negative', 'Positive', 'Negative'])
predicted = np.array(['Positive', 'Negative', 'Negative', 'Positive', 'Negative', 
                      'Positive', 'Positive', 'Negative', 'Positive', 'Negative'])

# Compute the confusion matrix
cm = confusion_matrix(actual, predicted, labels=['Positive', 'Negative'])

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Positive', 'Negative'], 
            yticklabels=['Positive', 'Negative'])
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('2-Class Confusion Matrix')
plt.show()
