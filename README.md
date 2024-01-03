<div align="center">
<h1 align="center">STYLE TRANSFER IN OPTIMAL TRANSPORT</h1>
  <h3 align="center">
    Project for OAA Course at International University - VNU
    <br />
    <br />
    <a href="https://github.com/meiskiet/OAA_StyleTransfer-/issues">Report Bug</a>
    ·
    <a href="https://github.com/meiskiet/OAA_StyleTransfer-/issues">Request Feature</a>
  </h3>

[![Forks][forks-shield]][forks-url] [![Issues][issues-shield]][issues-url]

</div>

<!-- About -->

# ABOUT

## 1. The team behind it

| No. |       Full Name        | Student's ID |              Email               |                       Roles                       | Contribution |
| :-: | :--------------------: | :----------: | :------------------------------: | :-----------------------------------------------: | :----------: |
|  1  |  Duong Nguyen Gia Khanh  | ITDSIU20100  | ITDSIU20100@student.hcmiu.edu.vn |       CODE AND DEMONSTRATION       |     25%      |
|  2  |    Nguyen Trung Kien     | ITDSIU20067  | ITDSIU20067@student.hcmiu.edu.vn |       SLIDES AND ALGORITHM        |     25%      |
|  3  |    Nguyen Hai Ngoc     | ITDSIU21057  | ITDSIU21057@student.hcmiu.edu.vn |             REPORT AND ALGORITHM             |     25%      |
|  4  | Hoang Tuan Kiet | ITDSIU21055  | ITDSIU21055@student.hcmiu.edu.vn | CODE AND DEMONSTRATION |     25%      |

## 2. The project we are working on

The project's goal is to introduce the optimal transport algorithm and its applications in style transfer.


<img src="src/components/assets/whole_project.png" alt="center">

<!-- METHODOLOGY -->

# METHODOLOGY

## 1. General method

In general, we have two inputs: a style image IS and a content image IC. The goal is to minimize the objective function below, using two inputs and gradient descent variant RMSprop.


<img src="src/components/assets/react_logo.jpg" alt="Objective function">

## 2. Style loss

### a. Relaxed earth movers distance

- []() Sorting selection

  <img src="src/components/assets/sort_selection.png" alt="sorting selection">



### b. Moment matching loss

Although lr can be a good term to transfer the structure of the source image to the target image, the feature vectors can be ignore by the cosine distance of lr. This results in a visual artifacts in the output, the typical example is over or under saturation. Therefore, the moment matching loss is apply to solve this problem:

<img src="src/components/assets/structure.png" alt="structure">


### c. Color matching loss

Color matching loss is used to ensure the output and the style image to have a similar palette. lp is computed using Euclidean distance with REMD between pixel colors in X(t) and Is. 

## 3. Content loss and user control

Equation to define content loss:


Where:
•	D_x is the pairwise cosine distance matrix of all feature vectors.
•	D_Ic is defined analogously for the content image.


- []()For the whole project:

  <img src="src/components/assets/fullUML.png" alt="fullUML">


<!-- INSTALLATION -->

# INSTALLATION

### Steps

1. Clone the repo
   ```sh
   git clone https://github.com/meiskiet/OAA_StyleTransfer-.git
   ```
2. Open in a programming IDE (supporting Python and its modules)
3. Open the Terminal
4. To install and run the project, users should run the code below in the
   terminal:

> npm install
>
> npm start

<!-- RESULT -->

# DEMO - RESULT



<!-- CONTRIBUTING -->

# CONTRIBUTING

Contributions are what make the open source community such an amazing place to
learn, inspire, and create. Any contributions you make are **greatly
appreciated**.

If you have a suggestion that would make this better, please fork the repo and
create a pull request. You can also simply open an issue with the tag
"enhancement". Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push develop feature/AmazingFeature`)
5. Open a Pull Request

<!-- CONTACT -->

# CONTACT

Hoang Tuan Kiet by **[Email HERE](ITDSIU21055@student.hcmiu.edu.vn)**

Project Link:
[https://github.com/meiskiet/OAA_StyleTransfer-](https://github.com/thanhhoann/DSA_wibudesu_sorting-simu)

<!-- ACKNOWLEDGMENTS -->

# ACKNOWLEDGEMENTS

We want to express our sincerest thanks to our lecturer and the people who have
helped us to achieve this project's goals:

- []() Assoc. Prof. Vo Thi Luu Phuong

- []() The README.md template from
  **[othneildrew](https://github.com/othneildrew/Best-README-Template)**

<!-- MARKDOWN LINKS & IMAGES -->

[forks-shield]: https://img.shields.io/github/forks/meiskiet/OAA_StyleTransfer-?style=for-the-badge
[forks-url]: https://github.com/meiskiet/OAA_StyleTransfer-/fork
[issues-shield]: https://img.shields.io/github/issues/meiskiet/OAA_StyleTransfer-?style=for-the-badge
[issues-url]: https://github.com/meiskiet/OAA_StyleTransfer-/issues