from quickai import ImageClassificationPredictor

predictions = ImageClassificationPredictor(
    "cars", 224, "00198.jpg", [
        'Acura Integra Type R 2001', 'Acura RL Sedan 2012'])
print(predictions)
