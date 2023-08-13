TEST_CASES = [
    # single image
    {
        'instruction': 'Create a slogan for the tags on the hummer focusing on the meaning of the word.',
        'images': [
            'https://visit-instruction-tuning.s3.amazonaws.com/visit_images/30_tested_skill_catchy_titles_4ca5b82782903f1c.png'
        ],
    },
    # multiple images
    {
        'instruction': 'In this task you will be provided with two individual images i.e., BEFORE and AFTER that are very similar, but with subtle differences. Please study them carefully to find out as many differences as you can for distinguishing AFTER image from the BEFORE image.',  
        'images': [
            "https://visit-instruction-tuning.s3.amazonaws.com/visit_images/474_spot_the_diff_67.png", 
            "https://visit-instruction-tuning.s3.amazonaws.com/visit_images/475_spot_the_diff_67_2.png"
        ],
    }
]