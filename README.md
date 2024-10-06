
# Automated G-code Generation for CNC Machining Using Generative AI

## Overview

This project aims to develop an AI-driven system that automates the generation of optimized G-code from 3D designs, addressing inefficiencies in manual G-code generation for CNC machining. The solution enhances precision, reduces human intervention, and streamlines the conversion of 3D models into machine-ready code.

## Key Features

- **Automated G-code Generation**: Utilizes Generative AI to automate the creation of optimized G-code from 3D designs.
- **Dimension Prediction**: Employs Convolutional Neural Networks (CNNs) for accurate prediction of object dimensions from 2D images.
- **Cloud-Ready Architecture**: Designed for scalability and modularity, making it suitable for large datasets and real-time processing.
- **User-Friendly Interface**: Offers a web-based platform for users to upload 2D designs and receive G-code efficiently.

## Problem Statement

Manual G-code generation in CNC machining is prone to errors and inefficiencies, requiring skilled operators. These challenges result in longer production times and increased costs. This project addresses these issues by providing an automated, AI-driven solution that enhances the manufacturing process.

## Technologies Used

- **Programming Languages**: Python
- **Frameworks**: Flask (for backend operations)
- **Deep Learning Libraries**: TensorFlow, Keras
- **Data Storage**: MongoDB
- **File Formatting**: XML (for structured G-code output)
- **Machine Learning Models**: Convolutional Neural Networks (CNNs), Generative AI models
- **CNC Simulation**: Validation of G-code efficiency using CNC simulators

## Getting Started

### Prerequisites

- Python 3.x
- Flask
- TensorFlow
- MongoDB

### Usage

1. Access the web interface by navigating to `http://localhost:5000` in your web browser.
2. Upload a 2D design image.
3. Receive the generated G-code for CNC machining.

## Results

- Achieved over 90% accuracy in dimension prediction.
- Generated G-code is 20% more efficient compared to traditional methods, reducing material waste and machining time.

## Future Work

- Enhance the model for multi-object recognition.
- Expand support for various CNC machine types.
- Improve the user interface for better user experience.

