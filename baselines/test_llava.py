from .llava_modeling import VisITLLaVA

model = VisITLLaVA({'model_path':"llava_13b/", "device":1})
print('Model loaded')

url = "https://visit-instruction-tuning.s3.amazonaws.com/visit_images/30_tested_skill_catchy_titles_4ca5b82782903f1c.png"
instruction = "Create a slogan for the tags on the hummer focusing on the meaning of the word."

url = "https://visit-instruction-tuning.s3.amazonaws.com/visit_images/886_kilogram_7.png"
instruction = "What does this shape look like? How would you segment different parts of the shape with same color?"

output = model.generate(instruction, [url])
print(output)