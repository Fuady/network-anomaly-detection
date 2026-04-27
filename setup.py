from setuptools import setup, find_packages
setup(
    name="network_anomaly_detection",
    version="1.0.0",
    description="Real-Time Network Anomaly Detection & Proactive Marketing",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=["numpy","pandas","scikit-learn","fastapi","pydantic","mlflow"],
)
