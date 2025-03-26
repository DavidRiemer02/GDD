from sdv.datasets.demo import download_demo

real_data, metadata = download_demo(
    modality='single_table',
    dataset_name='fake_hotel_guests'
)

from sdv.single_table import GaussianCopulaSynthesizer

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(real_data)

synthetic_data = synthesizer.sample(num_rows=2000)
synthetic_data['billing_address'] = synthetic_data['billing_address'].astype(str).str.replace(r'\s*\n\s*', ' ', regex=True).str.strip()
#Save the synthetic data
synthetic_data.to_csv('TestData/fakeData/sdv_hotel_data.csv')

from sdv.evaluation.single_table import run_diagnostic

diagnostic = run_diagnostic(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata
)

synthesizer.save('models/sdv/synthesizer.pkl')