# DL-CarClassificationWithFlask

- Install git 
```
sudo yum install git -y
```
- Clone the repository
```
 git clone https://github.com/zinebzannouti/CNN-CASE-STUDY
```
- install pip
```
curl -O https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user
```

- install flask
```
python3 -m pip install flask
```
- install tensorflow
```
python3 -m pip install tensorflow
```
- install PILLOW 

```
python3 -m pip install PILLOW
```
- go to DL-CarClassificationWithFlask
```
cd DL-CarClassificationWithFlask
```
- Create a dirrectory called static 
```
mkdir static
```
- copy your saved model from local to ec2 instance (open a new terminal)
```
 scp -i "training.pem" model.h5 ec2-user@ec2-3-90-21-179.compute-1.amazonaws.com:/home/ec2-user/DL-CarClassificationWithFlask
```

- Add the flaskapp.py that you have created :

```
 nano flaskapp.py
```
- boutton droit pour coller et ctrl+x -> y -> Entrer pour enregistrer

- Execute the flaskapp.py 

```
 python3 flaskapp.py
```

- Go to http://EC2-ip-address:5000/


