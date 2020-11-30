# Face recognition - 'It's Always Sunny in Philadelphia'

It's Always Sunny in Philadelphia is an american sitcom that follows the exploits of "The Gang" a group of narcissistic friends who run the Irish bar Paddy's Pub in South Philadelphia, Pennsylvania.

I used the <code>face-recognition</code> package in python to identify the main characters of the show. Obviously, face recognition has much more useful applications than identifying sitcom characters. But I used this small project, to understand how some common face recognition algorithms work.

For the face detection, I used HOG (Histogram of Oriented Gradients) to detect faces and encode the faces to vectors. I then used distance between the known faces and the detected faces to identify the character.

The results can be seen below. The algorithm correctly identifies and recognizes all the five characters 

<img src='videos/output_1.gif' >

# Tools used
+ Python 
+ OpenCV