# HW1 

> Apr 5, 2018

## Quick Start

### 1.1


### 1.2 

- 
- 

### 1.3

- Can network fit random labels?

- [__Number of parameters v.s. Generalization__](#)
	- __trainingMNIST_usingCNN.ipynb__ : 在 MNIST 上train pytorch CNN，產生不同parameters下對應的evaluation.
	- __trainingCIFAR10_usingKerasCNN.ipynb__ : 在 CIFAR10 上train Keras CNN，產生不同parameters下對應的evaluation.
	- __render.ipynb__ : 將上述兩個model的evaluation的pkl檔載入並作圖。
- [__Flatness v.s. Generalization - part1__](#)
	- __CNN.ipynb__ : 在MNIST上訓練兩個batch_size的model (50 & 1000)，並將其weight存成pkl檔。
	- __weight_linear_interpolation.ipynb__ : 將CNN.ipynb儲存的pkl讀入，並根據不同的alpha值依比例餵進customCNN中，並評估結果，最後將這些結果存到一個dict變數再存成pkl檔。
	- __render.ipynb__ : 根據載入的dict變數作圖。
- [__Flatness v.s. Generalization - part2__](#)
	- __CNN.ipynb__ : 在MNIST上選多個batch size並訓練, 評估。
	- __render.ipynb__ : 根據CNN.ipynb的評估結果作圖。
