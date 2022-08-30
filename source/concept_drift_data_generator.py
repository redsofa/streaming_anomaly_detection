from skmultiflow.data import ConceptDriftStream
from skmultiflow.data import AnomalySineGenerator, HyperplaneGenerator, AGRAWALGenerator,LEDGenerator
import pandas as pd
stream = AnomalySineGenerator(n_samples=10000, n_anomalies=900, n_contextual=0, replace=False)
num_of_data = 10000
data = stream.next_sample(num_of_data)
df = pd.DataFrame(data[0])
df['class'] = data[-1]
df.to_csv('max_test.csv', index=False)
print(df['class'].sum())
