# system-dl

The following files contain the implementation of ```Custom Linear layer``` and ```Lenet-300-100-10 Model```. 


```kernel.cu``` : custom matrix Multiplication and matrix Transpose kernel are written to implement ```Mul``` and ```Tpose``` API by using ```Codegen``` tool.

```custom_linear.py``` : Custom api ```Mul``` and ```Tpose``` have been used to implement a ```CustomLinearLayer``` by both implementing the forward calculation and backward calculation. 

```lenet_model.py```: The Lenet300-100-10 implementation using the ```CustomLinearLayer``` has been done here. 



