
Code for TallyQA  dataset published in AAAI 2018

For doing simple vs complex classification we used simpcomp.py file. You can load your dataset accordinlgy. TallyQA already
has a field that says whether the question is simple or complex so you don't need to run that file. For example, 
```
 {'answer': 4,
  'data_source': 'imported_genome',
  'image': 'VG_100K_2/2410408.jpg',
  'image_id': 92410408,
  'issimple': False, ## this entry here
  'question': 'How many headlights does the black bus have?',
  'question_id': 30095774}
```
The main paper uses the `RN_BG_OG_embd` model with the reported hyper-parameters in this repo.
There are other types of Relational Models avaialabe in the repo which we tried for our project.

To Run the code as:
```python main.py --model RN_OG_embd ```

Don't forget to edit config.py to point to appropriate  locations for files and features.
Please, feel free to ask if you have any other questions.
