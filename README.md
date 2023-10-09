# system-dl

The following files contain the implementation of ```Custom Linear layer``` and ```Lenet-300-100-10 Model```. 


```kernel.cu``` : custom matrix Multiplication and matrix Transpose kernel are written to implement ```Mul``` and ```Tpose``` API by using ```Codegen``` tool.

```custom_linear.py``` : Custom api ```Mul``` and ```Tpose``` have been used to implement a ```CustomLinearLayer``` by both implementing the forward calculation and backward calculation. 

```lenet_model.py```: The Lenet300-100-10 implementation on MNIST dataset using the ```CustomLinearLayer``` has been done here. 

the ```test_apis.py``` script conatins code which has been used to test the kernel operation and matrix dimension correctly.

```result.txt``` file contains the train and test accuracy result (for 200 epoch) of  ```Lenet-300-100-10 Model``` 



