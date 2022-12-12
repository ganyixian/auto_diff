# Team04 Final Project: AutoDiff Package
 
[![.github/workflows/test.yml](https://code.harvard.edu/CS107/team04/actions/workflows/test.yml/badge.svg)](https://code.harvard.edu/CS107/team04/actions/workflows/test.yml)
[![.github/workflows/coverage.yml](https://code.harvard.edu/CS107/team04/actions/workflows/coverage.yml/badge.svg)](https://code.harvard.edu/CS107/team04/actions/workflows/coverage.yml)

## Team 04
Yixian Gan, Siyao Li, Li Yao, Ting Cheng, Haitian Liu

## Brief Introduction
This is the final project for Harvard CS107/AC207 class in 2022 Fall. The package was designed to calculate the first-order derivative of any user-defined functions in the direction of a given seed vector. Both forward and reverse modes are supported. 

For the detailed instruction, please refer to [documentation.ipynb](https://code.harvard.edu/CS107/team04/blob/main/docs/documentation.ipynb) in /docs. 

## Broader Impact 
Our package was designed to compute the first-order derivative of any given function using both forward and reverse mode of automatic differentiation. Compared to symbolic differentiation and numerical differentiation, automatic differentiation provides a less costly but efficient way of calculating derivatives while maintaining machine precision. Automatic differentiation is widely used in machine learning, data science, audio signal processing and many other fields. We believe the nature of automatic differentiation would be helpful for these large projects, which require a large amount of computing power.
Also, our project can also serve an educational purpose. For example, when students are learning how to calculate derivatives by hands, our package could help the instructor quickly derive the desired derivative, and help students check their attempted solutions quickly. But a potential scenario could be that students skip the process of understanding the concepts of derivative and learning how to solve limits but copy the returned results to their homework. 


## Software Inclusivity 

AutoDiff was designed to welcome users and contributors from all backgrounds. The package was developed based on the key principles of python community: mutual respect, tolerance and encouragement. Throughout the process of development, we made every effort to create an inclusive and user-friendly package including but not limited to asking TF for feedback, writing doc strings, and providing sufficient documentation. Every team member contributed to review pull requests, provide feedback to each other, and approve pull requests. Admittedly, the package was written in English and Python, but there will be opportunities to localize the package in the future. We used the MIT license and planned to release the package to the open source community so anyone who has experience in another language will have the opportunity to rewrite and translate the package. 
