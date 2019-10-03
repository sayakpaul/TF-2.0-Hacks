In this mini-project, I present how eager execution can really be helpful in speeding up the model training process.

Steps I followed to conduct the experiments:
- I maintained the exact same environment, model configuration, dataset (FashionMNIST) for the experiments. I only changed the TensorFlow versions. 
- I ran thorough profiling to check what really causes execution in TensorFlow 1.14 to be slow and I found out it was _Sessions_.

Apart from these, I used [Weights and Biases](https://wandb.com) to log the CPU usage and memory footprints of the experiments. I was amazed to find out that TensorFlow 2.0 was much more performant in terms of CPU usage as well. Here are some snaps:

![](https://i.ibb.co/D9VcssK/Screen-Shot-2019-10-03-at-12-12-03-PM.png)

![](https://i.ibb.co/5vPTfwj/Screen-Shot-2019-10-03-at-12-13-21-PM.png)

