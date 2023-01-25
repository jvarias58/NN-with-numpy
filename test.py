val = 0
for i in range(1000):
    index = i
    img = x_test[:,index].reshape(784, 1)
    
    #img = np.asarray(Image.open('out.png')).reshape(784,1)
    
    #plt.imshow(img.reshape(28, 28), cmap="Greys")
    
    a = 1 / (1 + np.exp(-(b1 + np.dot(w1, img))))
    
    b = 1 / (1 + np.exp(-(b2 + np.dot(w2, a))))
    
    c = 1 / (1 + np.exp(-(b3 + np.dot(w3, b))))
    
    #print(c.argmax(axis=0))
    if (c.argmax(axis=0) == labels2[:,i].argmax(axis=0)):
        val += 1
print(val/1000)