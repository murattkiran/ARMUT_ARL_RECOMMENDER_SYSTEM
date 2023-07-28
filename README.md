# Armut Association Rule Based Recommender System

![Armut Logo](https://user-images.githubusercontent.com/84645968/217642487-de13b7cd-2fa8-4759-9dc1-af6a31877e8e.png)

## Business Problem

Armut is Turkey's largest online service platform that connects service providers with customers seeking services. It enables users to easily access services like cleaning, renovation, and moving through their computers or smartphones. In this project, we aim to create a product recommendation system for Armut's users who have received services. By utilizing Association Rule Learning, we will help users discover better services and suggest new services that align with their preferences.

## Dataset

The dataset used for this project contains information about services received by Armut's customers and the categories of these services. Additionally, it includes the date and time when the services were purchased.

Main columns in the dataset:

- **UserId**: Customer number.
- **ServiceId**: Anonymized service number. The same ServiceId may represent different services under different categories.
- **CategoryId**: Anonymized category number. Each CategoryId corresponds to a different service category.
- **CreateDate**: The date when the service was purchased.

## Project Objective

The main objective of this project is to learn association rules based on users' past service purchases and provide personalized service recommendations. By analyzing the association rules derived from users' preferences and historical purchases, we aim to suggest new services that match their interests and preferences.
