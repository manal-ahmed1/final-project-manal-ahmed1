[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/j5u2qjgo)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=22969664)
> [!IMPORTANT]
> Replace this README.md with your own content and use this repo for your project

# Where is Waldo 

The “Where’s Waldo?” picture game is a task where the goal is to locate the titular character "Waldo" within a densely populated image filled with distracting patterns, colors, and similar-looking characters. A CNN's approach to this problem is by learning hierarchical visual features, starting from simple edges and textures in early layers to more complex shapes and patterns in deeper layers. Through training on labeled examples of close ups of both "waldo" and "not waldo" images, the network learns to distinguish Waldo’s unique attributes, such as his red-and-white striped shirt and hat, from the "not waldo" surrandings. This CNN should be able to find Waldo in larger images based on snapshots of Waldo and his surrandings.  

Data pulled from Aleksey Bilogur's "Where's Waldo" on Kaggle:
https://www.kaggle.com/datasets/residentmario/wheres-waldo/data

## How to Run the Code (Dependencies + Setup)

This project is implemented in **Python 3** using **TensorFlow + Keras** for the CNN, with supporting libraries for data handling and visualization.

#### 1) Environment setup
1. Create & activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate     # macOS
   ```

2. Install required dependencies:
   ```bash
   pip install tensorflow matplotlib numpy pandas pillow
   ```

#### 2) Dataset layout
The notebook expects the dataset to be organized like this:
```
data/
  train/
    waldo/       # positive examples (contains Waldo)
    nowaldo/     # negative examples (no Waldo)
  test/          # unlabeled images to evaluate generalization
```

#### 3) Running the notebook
1. Start Jupyter:
   ```bash
   jupyter notebook neural_network.ipynb
   ```
2. Execute cells **top-to-bottom** (especially the data loading sections) so the dataset and model are initialized correctly.

#### 4) Notes / Troubleshooting
- **Images are resized to 224×224
- Do not double normalize
- For MobileNetV2:
  - use preprocess_input
  - do NOT divide by 255 again
  - Since  ImageNet weights failed to download (SSL issue), I used weights=None

## Dataset Description

**Source:** "Where's Waldo" dataset from Kaggle (Aleksey Bilogur)

**Dataset Purpose:** A binary image classification task to detect whether a given image patch contains Waldo or not.

**Classes:**
- **Waldo (Class 1):** Image crops containing Waldo's distinctive red-and-white striped shirt and hat
- **Not Waldo (Class 0):** Image crops of background scenery, other characters, or distracting patterns

**Dataset Split:**
- **Training Set:** 5,449 labeled images total in train folder
  - Waldo images: 96 samples
  - Not-Waldo images: 5,353 samples
  - Note: Highly imbalanced dataset (1.8% positive class)
- **Test Set:** 19 unlabeled images (for evaluating generalization)

**Input Features:**
- Raw pixel values from RGB color images
- Original images have varying dimensions:
  - Waldo patches: 64×64 pixels
  - Not-Waldo patches: typically 256×256 pixels
- All images resized to uniform dimensions of **224×224×3** for model training, where:
  - 224 = image height (pixels)
  - 224 = image width (pixels)  
  - 3 = RGB color channels

**Outcome Variable:**
- **Binary label:**
  - 1 = Waldo present
  - 0 = Waldo not present

**Data Format:**
- Stored in directory structure: `data/train/{waldo, nowaldo}/` and `data/test/`
- Image format: JPEG files
- Pixel Processing:
  - Standard CNN--> normalized to [0,1]
  - MobileNetV2: uses preprocess_input instead

## Decisions made along the way and trade-offs

Several decisions were made during development:

1. We used a CNN because the task is image-based and CNNs are designed to learn spatial patterns effectively.

2. Images were resized to a fixed size for consistency. This simplifies training but may remove fine detail, which is important since Waldo is small.

3. We increased image size from 128×128 to 224×224 to preserve detail, at the cost of increased training time. 

4. We deepened the CNN by adding multiple convolution and pooling layers. This allowed the network to learn more complex visual patterns, but it also made the model more computationally expensive.

5. We trained the model without a formal validation split in the final version to simplify the workflow and use all labeled training data. From my undersanding, the limitations that come with this decision is that we cannot monitor overfitting as carefully.

6. We considered data augmentation to improve generalization, but depending on the final version of the notebook, this may not have been fully integrated into training. As a result, performance may be limited on more varied test examples.

7. We used binary classification instead of object detection, meaning the model predicts presence of Waldo but not location.

8. A major challenge was extreme class imbalance:
 - 96 Waldo vs 5,353 not-Waldo images
 - This caused misleading high accuracy (~98%)
 - The model initially learned to always predict "not Waldo"

 To address this:
 - we introduced class weights
 - explored balancing strategies (e.g., oversampling)

9. A CNN trained from scratch failed due to limited Waldo examples. We switched to transfer learning (MobileNetV2), which significantly improved feature learning and generalization.

## Final Project

The final assignment for this class is a multi-week project. The project is self-driven but the expectation is that you will work in groups to demonstrate your ability to do something original with your newfound pythonic abilities. 

It’s up to you, but some suggestions include:

- Testing ideas towards a Capstone project
- Replicating and extending analysis done in a published paper
- Working with an existing codebase/model to apply an interesting ML method
- Performing a novel analysis on a dataset

## Expectations

- This should involve original work from your team (size of group: anywhere between 1-38)
- Level of effort should be ~2-3 weeks worth of work
- Submission will be a repository including:
    - Code: your own and perhaps from an existing project
    - Documentation:
        - Overview of the problem
        - Description of the dataset you used (input features, outcome, dimensions, etc)
        - How to run the code (dependencies, etc.)
        - Decisions made along the way, including trade-offs (e.g., cut X for time so our solution may lack Y)
        - Example output (what does it do?)
        - Citations (data, code, papers)
    - Short-form presentation (slides and notes or video, not during lecture):
        - 10 minutes with fellow classmates as the target audience
        - Problem statement
        - Existing work you pulled from
        - Your contribution
        - Tools/methods used
        - Issues overcome along the way
        - PPT or document style OK

# Inspiration

## Data & code

- [The Incredible PyTorch](https://github.com/ritchieng/the-incredible-pytorch)
    
    > List of papers, code, examples using PyTorch
    > 
- [Kaggle](https://www.kaggle.com)
    
    > ML competition/collaboration site
    > 
- [Keras code examples](https://keras.io/examples/)
    
    > Official examples of implementation using Google’s TensorFlow Keras
    > 
- [PLOS papers with available data](https://journals.plos.org/plosone/search?q=data_availability%3A(osf.io%20OR%20github%20OR%20dryad%20OR%20figshare)&page=1)
    
    > Searching PLOS for keywords likely to have available data, refine further to get topics interesting to you
    > 
- [PLOS recommended repositories](https://journals.plos.org/plosone/s/recommended-repositories) (data, code, and sometimes both)
    
    > Lots here, mostly data repositories
    > 
- [The Pudding](https://www.pudding.cool)
    
    > Visual essays with data
    >
