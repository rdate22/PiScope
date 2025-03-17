# SentriLock™ - Secure Facial Recognition Access Control System

![Capstone Group Picture](/grouppic.JPG)
Embedded Systems capstone project UW Seattle

SentriLock™ is an advanced access control system that integrates real-time facial recognition with UART-based communication for seamless and secure entry management. Designed for efficiency, security, and ease of use, SentriLock™ replaces traditional key and RFID-based authentication methods with biometric verification, ensuring only authorized users can gain access.

### Features

Real-time Facial Recognition – Uses OpenCV and Dlib to detect and authenticate users in under 200ms.
Embedded System Integration – Communicates with an STM32 microcontroller via UART to trigger door unlocking mechanisms.
Secure and Private – Local processing ensures that no facial data is stored in the cloud, protecting user privacy.
Web-Based User Management – Flask-powered interface allows for remote user enrollment and access monitoring.
Optimized Performance – Multithreaded processing improves recognition speed while reducing CPU load.
Scalable and Customizable – Designed to support additional authentication layers such as PIN codes or NFC verification.

### Technologies Used

Python, OpenCV, Dlib – Facial recognition and image processing.
Flask – Web-based user management and API.
PySerial – UART communication with STM32.
C/C++ for STM32 – Embedded firmware for access control.
Raspberry Pi 4 – Handles facial recognition and system processing.

### Contributors

**Embedded Systems Engineer** - Brayden Lam
**Embedded Systems Engineer** - Jackson O’Neill
**Project Manager & Software Engineer: Embedded Systems**  - Jerick Ilano
**Software Engineer: Machine Learning** - Matthew Nochi
**Software Engineer: Full-Stack** - Rohan Date

This repo contains the python code that runs on the raspberry pi
STM code repo: https://github.com/braylam/SentriLock_STM
