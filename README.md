# Final-Year-Research-
💎 GemAI – Automated Gemstone Analysis System

GemAI is an AI-powered system designed to automate and improve gemstone evaluation in the Sri Lankan gem industry. It focuses on accurate identification, clarity analysis, cutting optimization, and gemstone classification using computer vision, machine learning, and custom-built hardware devices.
The system mainly targets Yellow Sapphire and Zircon, which are commonly misidentified and difficult to evaluate through manual inspection.

🚀 Project Objectives

Reduce human dependency in gemstone evaluation
Improve accuracy and consistency in gem identification
Minimize material loss during cutting
Provide a low-cost and user-friendly solution for gem traders and cutters

🧩 System Components

GemAI consists of four main AI-based components:

1️⃣ Real vs Synthetic Gemstone Identification
(Polarized Light Analysis)
This component identifies whether a gemstone is natural or synthetic using a polarized light–based testing method.

A custom-built hardware setup is used, which includes:
- Two polarized sheets
- A white LED light source
- A rotating gemstone platform

During testing, the gemstone is placed between the polarized sheets while the top sheet rotates. This causes visible changes in light and color patterns, which differ between natural and synthetic gemstones.
Images captured during this rotation are processed using image analysis and machine learning models to classify the gemstone accurately. This approach provides a consistent, low-cost, and reliable alternative to expert-based manual testing.

2️⃣ Clarity Detection Based on Crack Analysis
This component evaluates the clarity of rough gemstones by detecting cracks and internal flaws.
A custom IoT-based image capturing device is used, consisting of:

- Arduino board
- Servo motor
- Camera
- Controlled lighting setup

The gemstone is rotated at 10-degree intervals, capturing 36 images per gemstone. These images are analyzed using a YOLOv8 object detection model, which is trained to identify cracks in rough Yellow Sapphire and Zircon stones.

This system:
- Reduces human subjectivity
- Improves crack detection accuracy
- Helps gem cutters avoid cutting flawed stones
- Prevents unnecessary material loss

3️⃣ Cutting-Dimension Optimization

This component suggests the best cutting shape and optimal dimensions for a gemstone.
Using the 36 clarity-detected images:
The gemstone’s length, width, and usable area are calculated
Crack and flaw regions are removed
The remaining usable gemstone region is identified
Machine learning techniques are then used to determine the most suitable cutting shape and dimensions. A 3D gemstone model is generated to visually represent the recommended cut before physical cutting.
This helps:
- Minimize wastage
- Improve cutting accuracy
- Increase final gemstone value

4️⃣ Gem Type Identification & Color Detection

This component identifies the type and color of the gemstone.
A custom-built spectrometer scans the light spectrum of the gemstone. The spectral data is processed using a machine learning model to classify whether the stone is Yellow Sapphire or Zircon.
At the same time, a high-resolution image is captured under controlled lighting and analyzed using color detection algorithms to determine the gemstone’s color tone, such as:
* Fancy Yellow
* Golden Yellow
* Medium Yellow

This contributes to the gemstone’s overall grading and market value.

🛠 Technologies Used

* Python
* YOLOv8 (Object Detection)
* Machine Learning
* OpenCV & Image Processing
* Arduino & IoT Hardware
* Spectrometer
* 3D Visualization

🌟 Key Benefits

* Reduces reliance on human experts
* Increases gemstone evaluation accuracy
* Supports better cutting decisions
* Low-cost and scalable solution
* Designed for the Sri Lankan gem industry

📌 Target Users

* Gem traders
* Gem cutters
* Gemological laboratories
* Jewelry manufacturers

  
