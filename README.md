# ReLU-vs-Abs-activation
Comparing the performance between ReLU and Absolute function as the activation function in Neural Networks
This is a very simple work, I did this because someone asked me in an interview.

Our hypothesis about the major problem of Absolute function is that it's not monotonic. Recall that the common activation functions we used like Sigmoid, Tanh, ReLU, are all monotonic in their domains. The reason why we want a monotonic activation function is that it preserves convexity of the input. Though modern Neural Networks are mostly non-convex in their loss landscape, it would be a bad idea to incur more non-convexity than it could be, which would make the optimization even harder.

I did a simple experiment comparing Absolute function and ReLU function as the activation function in the classification problem using Resnet20 on Cifar10. I ran 200 epochs with a batch size of 128 and optimized with SGD with momentum. The attached result attests to our intuition that NNs with Absolute activation are harder to optimize. You can see there is a 8% accuracy drop for the Absolute Function.
