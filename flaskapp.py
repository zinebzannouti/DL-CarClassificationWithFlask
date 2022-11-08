from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

from tensorflow.keras import backend,layers
app=Flask(__name__,template_folder='templates')
class FixedDropout(layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape=backend.shape(inputs)
        noise_shape=[symbolic_shape[axis] if shape is None else shape
                     for axis,shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)
model = load_model('C:/Users/User/Downloads/Efficientnet_model.h5',custom_objects={'FixedDropout':FixedDropout(rate=0.4)})

model.make_predict_function()
dic = {0:'AM General Hummer SUV 2000',
1:'Acura Integra Type R 2001',
2:'Acura RL Sedan 2012',
3:'Acura TL Sedan 2012',
4:'Acura TL Type-S 2008',
5:'Acura TSX Sedan 2012',
6:'Acura ZDX Hatchback 2012',
7:'Aston Martin V8 Vantage Convertible 2012',
8:'Aston Martin V8 Vantage Coupe 2012',
9:'Aston Martin Virage Convertible 2012',
10:'Aston Martin Virage Coupe 2012',
11:'Audi 100 Sedan 1994',
12:'Audi 100 Wagon 1994',
13:'Audi A5 Coupe 2012',
14:'Audi R8 Coupe 2012',
15:'Audi RS 4 Convertible 2008',
16:'Audi S4 Sedan 2007',
17:'Audi S4 Sedan 2012',
18:'Audi S5 Convertible 2012',
19:'Audi S5 Coupe 2012',
20:'Audi S6 Sedan 2011',
21:'Audi TT Hatchback 2011',
22:'Audi TT RS Coupe 2012',
23:'Audi TTS Coupe 2012',
24:'Audi V8 Sedan 1994',
25:'BMW 1 Series Convertible 2012',
26:'BMW 1 Series Coupe 2012',
27:'BMW 3 Series Sedan 2012',
28:'BMW 3 Series Wagon 2012',
29:'BMW 6 Series Convertible 2007',
30:'BMW ActiveHybrid 5 Sedan 2012',
31:'BMW M3 Coupe 2012',
32:'BMW M5 Sedan 2010',
33:'BMW M6 Convertible 2010',
34:'BMW X3 SUV 2012',
35:'BMW X5 SUV 2007',
36:'BMW X6 SUV 2012',
37:'BMW Z4 Convertible 2012',
38:'Bentley Arnage Sedan 2009',
39:'Bentley Continental Flying Spur Sedan 2007',
40:'Bentley Continental GT Coupe 2007',
41:'Bentley Continental GT Coupe 2012',
42:'Bentley Continental Supersports Conv. Convertible 2012',
43:'Bentley Mulsanne Sedan 2011',
44:'Bugatti Veyron 16.4 Convertible 2009',
45:'Bugatti Veyron 16.4 Coupe 2009',
46:'Buick Enclave SUV 2012',
47:'Buick Rainier SUV 2007',
48:'Buick Regal GS 2012',
49:'Buick Verano Sedan 2012',
50:'Cadillac CTS-V Sedan 2012',
51:'Cadillac Escalade EXT Crew Cab 2007',
52:'Cadillac SRX SUV 2012',
53:'Chevrolet Avalanche Crew Cab 2012',
54:'Chevrolet Camaro Convertible 2012',
55:'Chevrolet Cobalt SS 2010',
56:'Chevrolet Corvette Convertible 2012',
57:'Chevrolet Corvette Ron Fellows Edition Z06 2007',
58:'Chevrolet Corvette ZR1 2012',
59:'Chevrolet Express Cargo Van 2007',
60:'Chevrolet Express Van 2007',
61:'Chevrolet HHR SS 2010',
62:'Chevrolet Impala Sedan 2007',
63:'Chevrolet Malibu Hybrid Sedan 2010',
64:'Chevrolet Malibu Sedan 2007',
65:'Chevrolet Monte Carlo Coupe 2007',
66:'Chevrolet Silverado 1500 Classic Extended Cab 2007',
67:'Chevrolet Silverado 1500 Extended Cab 2012',
68:'Chevrolet Silverado 1500 Hybrid Crew Cab 2012',
69:'Chevrolet Silverado 1500 Regular Cab 2012',
70:'Chevrolet Silverado 2500HD Regular Cab 2012',
71:'Chevrolet Sonic Sedan 2012',
72:'Chevrolet Tahoe Hybrid SUV 2012',
73:'Chevrolet TrailBlazer SS 2009',
74:'Chevrolet Traverse SUV 2012',
75:'Chrysler 300 SRT-8 2010',
76:'Chrysler Aspen SUV 2009',
77:'Chrysler Crossfire Convertible 2008',
78:'Chrysler PT Cruiser Convertible 2008',
79:'Chrysler Sebring Convertible 2010',
80:'Chrysler Town and Country Minivan 2012',
81:'Daewoo Nubira Wagon 2002',
82:'Dodge Caliber Wagon 2007',
83:'Dodge Caliber Wagon 2012',
84:'Dodge Caravan Minivan 1997',
85:'Dodge Challenger SRT8 2011',
86:'Dodge Charger SRT-8 2009',
87:'Dodge Charger Sedan 2012',
88:'Dodge Dakota Club Cab 2007',
89:'Dodge Dakota Crew Cab 2010',
90:'Dodge Durango SUV 2007',
91:'Dodge Durango SUV 2012',
92:'Dodge Journey SUV 2012',
93:'Dodge Magnum Wagon 2008',
94:'Dodge Ram Pickup 3500 Crew Cab 2010',
95:'Dodge Ram Pickup 3500 Quad Cab 2009',
96:'Dodge Sprinter Cargo Van 2009',
97:'Eagle Talon Hatchback 1998',
98:'FIAT 500 Abarth 2012',
99:'FIAT 500 Convertible 2012',
100:'Ferrari 458 Italia Convertible 2012',
101:'Ferrari 458 Italia Coupe 2012',
102:'Ferrari California Convertible 2012',
103:'Ferrari FF Coupe 2012',
104:'Fisker Karma Sedan 2012',
105:'Ford E-Series Wagon Van 2012',
106:'Ford Edge SUV 2012',
107:'Ford Expedition EL SUV 2009',
108:'Ford F-150 Regular Cab 2007',
109:'Ford F-150 Regular Cab 2012',
110:'Ford F-450 Super Duty Crew Cab 2012',
111:'Ford Fiesta Sedan 2012',
112:'Ford Focus Sedan 2007',
113:'Ford Freestar Minivan 2007',
114:'Ford GT Coupe 2006',
115:'Ford Mustang Convertible 2007',
116:'Ford Ranger SuperCab 2011',
117:'GMC Acadia SUV 2012',
118:'GMC Canyon Extended Cab 2012',
119:'GMC Savana Van 2012',
120:'GMC Terrain SUV 2012',
121:'GMC Yukon Hybrid SUV 2012',
122:'Geo Metro Convertible 1993',
123:'HUMMER H2 SUT Crew Cab 2009',
124:'HUMMER H3T Crew Cab 2010',
125:'Honda Accord Coupe 2012',
126:'Honda Accord Sedan 2012',
127:'Honda Odyssey Minivan 2007',
128:'Honda Odyssey Minivan 2012',
129:'Hyundai Accent Sedan 2012',
130:'Hyundai Azera Sedan 2012',
131:'Hyundai Elantra Sedan 2007',
132:'Hyundai Elantra Touring Hatchback 2012',
133:'Hyundai Genesis Sedan 2012',
134:'Hyundai Santa Fe SUV 2012',
135:'Hyundai Sonata Hybrid Sedan 2012',
136:'Hyundai Sonata Sedan 2012',
137:'Hyundai Tucson SUV 2012',
138:'Hyundai Veloster Hatchback 2012',
139:'Hyundai Veracruz SUV 2012',
140:'Infiniti G Coupe IPL 2012',
141:'Infiniti QX56 SUV 2011',
142:'Isuzu Ascender SUV 2008',
143:'Jaguar XK XKR 2012',
144:'Jeep Compass SUV 2012',
145:'Jeep Grand Cherokee SUV 2012',
146:'Jeep Liberty SUV 2012',
147:'Jeep Patriot SUV 2012',
148:'Jeep Wrangler SUV 2012',
149:'Lamborghini Aventador Coupe 2012',
150:'Lamborghini Diablo Coupe 2001',
151:'Lamborghini Gallardo LP 570-4 Superleggera 2012',
152:'Lamborghini Reventon Coupe 2008',
153:'Land Rover LR2 SUV 2012',
154:'Land Rover Range Rover SUV 2012',
155:'Lincoln Town Car Sedan 2011',
156:'MINI Cooper Roadster Convertible 2012',
157:'Maybach Landaulet Convertible 2012',
158:'Mazda Tribute SUV 2011',
159:'McLaren MP4-12C Coupe 2012',
160:'Mercedes-Benz 300-Class Convertible 1993',
161:'Mercedes-Benz C-Class Sedan 2012',
162:'Mercedes-Benz E-Class Sedan 2012',
163:'Mercedes-Benz S-Class Sedan 2012',
164:'Mercedes-Benz SL-Class Coupe 2009',
165:'Mercedes-Benz Sprinter Van 2012',
166:'Mitsubishi Lancer Sedan 2012',
167:'Nissan 240SX Coupe 1998',
168:'Nissan Juke Hatchback 2012',
169:'Nissan Leaf Hatchback 2012',
170:'Nissan NV Passenger Van 2012',
171:'Plymouth Neon Coupe 1999',
172:'Porsche Panamera Sedan 2012',
173:'Ram C-V Cargo Van Minivan 2012',
174:'Rolls-Royce Ghost Sedan 2012',
175:'Rolls-Royce Phantom Drophead Coupe Convertible 2012',
176:'Rolls-Royce Phantom Sedan 2012',
177:'Scion xD Hatchback 2012',
178:'Spyker C8 Convertible 2009',
179:'Spyker C8 Coupe 2009',
180:'Suzuki Aerio Sedan 2007',
181:'Suzuki Kizashi Sedan 2012',
182:'Suzuki SX4 Hatchback 2012',
183:'Suzuki SX4 Sedan 2012',
184:'Tesla Model S Sedan 2012',
185:'Toyota 4Runner SUV 2012',
186:'Toyota Camry Sedan 2012',
187:'Toyota Corolla Sedan 2012',
188:'Toyota Sequoia SUV 2012',
189:'Volkswagen Beetle Hatchback 2012',
190:'Volkswagen Golf Hatchback 1991',
191:'Volkswagen Golf Hatchback 2012',
192:'Volvo 240 Sedan 1993',
193:'Volvo C30 Hatchback 2012',
194:'Volvo XC90 SUV 2007',
195:'smart fortwo Convertible 2012'}
def predict_label(img_path):
	i = image.load_img(img_path, target_size=(240,240))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 240,240,3)
	p = model.predict(i)
	in_max=np.where(p[0]==np.max(p))
	return dic[in_max[0][0]]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':           
		img = request.files['my_image']

		img_path ="static/" +img.filename
		img.save(img_path)
		a=img.filename     

		m = predict_label(img_path)

	return render_template("index.html", prediction = m, a = a)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
