from setuptools import setup, find_packages

setup(
    name="lets_dance",
    version="0.1.0",
    description="On-device AI dance scoring game using MediaPipe pose estimation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "mediapipe>=0.10.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "pygame>=2.5.0",
        "pyyaml>=6.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "scratch-runtime": [
            "tflite-runtime>=2.14.0",
        ],
        "scratch-train": [
            "tensorflow==2.16.2",
            "jax==0.4.26",
            "jaxlib==0.4.26",
            "ipykernel>=6.29.0",
        ],
        "mpose-pretrain": [
            "mpose==1.2",
            "tensorflow==2.16.2",
            "jax==0.4.26",
            "jaxlib==0.4.26",
            "ipykernel>=6.29.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lets-dance=main:main",
        ],
    },
)
