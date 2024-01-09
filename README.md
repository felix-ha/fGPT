# fGPT

> A language model trained from scratch

Test the model here: https://felixgpt.streamlit.app
	

# Overview

The model architecture is a GPT-2 style encoder only transformer (`dim_embedings=768`, `n_layers=4` and `n_blocks=4`) with context of 1000 tokens and has 38M parameters.

The model was trained on the [tiny stories](https://arxiv.org/abs/2305.07759) dataset using spaCy englisch words tokenizer. 

It is inspired from Andrej Karpathy's  [nanoGPT](https://github.com/karpathy/nanoGPT) and his series [nn form zero to hero](https://github.com/karpathy/nn-zero-to-hero).

# Current status

## ToDos

* Improve training, initialization, hyperparmeter etc. At the mome I just a basic AdamW optimizer with nothing more tested and investigated.

## Results

* The training dataset consits of 2,717,699 stories. The model that is deployed right now was trained for epochs 
* I only use 10k most common tokens instead of 49959 unique tokens that are found in the whole dataset.

Here are the prompts that were evaluated in chapter 4.4 of the original paper. 

Only the first part of the generated story is rated. 

Prompt | fGPT | 33M 4 layers of original paper
-------- | -------- | --------
Alice was so tired when she got back home so she went | to sleep. ✅ | straight to bed. ✅
Jack and Lily saw a rainbow after a rainy day. They were amazed by the colors. Jack said, "Look, Lily. A rainbow has   | " a rainbow on it." ❌ | red, orange, yellow, green, blue, and purple! ✅
Jack and Lily liked to watch the moon at night. They noticed that the moon changed its shape every night. Sometimes the moon was big and round, and sometimes it was   | big.  ✅   | small and thin. ✅
Jack wanted to read a book, so he went to   | the library.  ✅ | the library. ✅
"Can cows fly?", Alice asked her mother.   | "Yes, sweetheart. Just flap your arms and try to catch the sky." ❌   | "No, they can't fly," her mother said. ✅
"What do birds like to eat?", Tom asked his mother.   |"Mom, what does that sound like?" ❌  | His mother smiled and said, "They like to eat worms and bugs." ✅
"What language do they speak in France?", Tom asked his mother   | "I don't know,"" said his mother. ❌  | "They speak French," his mother replied. ✅
If I throw a ball up in the air, eventually it will  |  come down".  ✅ | come down." ✅
It was winter and cold outside so his mother told him, "You should   |  wear your warm coat. ✅  | wear your warm coat so you don't get cold. ✅
Lily likes cats and dogs. She asked her mom for a dog and her mom said no, so instead she asked   |  for a bone.  ❌  | her dad for a cat. ✅
Jack told Mary, "If you give me your banana, I’ll give you my apple". Mary gave Jack her Banana so   | he could go to the market. ❌ | he could give her the apple. ✅
On weekends Jack went to visit his grandmother whereas on weekdays he would go to school. Last weekend, when Jack was on his way to   | school ❌   | his grandmother's house ✅
Lily and Ben were having an argument. Ben said that cake is much better than ice cream and Lily said that   |  Lily said that the cake was too hot and she could hurt her.  ❌  | ice cream is better than cake ✅
Lily and Ben are having an argument. They are trying to decide between the park and the swimming pool. Ben says, "I want to go to the park". Lily says   |  "No, I want to go to the park"  ❌ | , "No, I want to go to the pool". ✅
Jack's mother was not home, and his father was at home. When Jack came home, he said hello to   | his mother. ❌ | his mother, but she didn't answer. Jack was confused. ✅
Lily doesn't like swimming. When her father wants to take her to the swimming pool, she says   |   it is too hot. ❌ | , "No, I don’t want to go. I want to stay here." ✅
Both Ben and Lily wanted cake. Father said that there was only one piece of cake left. They   |  were very sad. ✅  | started to fight over the cake. They pulled and pushed and shouted. ✅
Ben went to visit Lily in her house, but she was not at home. Ben knocked on the door,   | and a big, hairy dog named Max answered, ❌  | but no one answered. He knocked again and again, but still  no one came. ✅
"Hi Jane, have you seen Alice? I cannot find her anywhere", said Jack.   |  "Enjoy," said Jane. ❌  | Jane smiled and said, "Don' t worry  Jack, I'll help you find her". Jack and Jane looked around the park, but they couldn't find Alice. ✅
Max had two dogs. One was white and the other was black. Max walked up the street and saw a kid with a dog. He told the kid, "I see you have a Brown dog. I also have | one. ❌  | two dogs. ✅
Anne had a piece of candy in her left pocket and a piece of chocolate in her right pocket. Annes mom asked her, "Anne, what is that you have in your left pocket?"   |   Anne said, "I have candy in my pocket. It's a special candy that you can eat."  ✅ | Anne replied, "It's a piece of candy, Mommy. It's so yummy!" ✅
Alice had both an apple and a carrot in her bag. She took the apple out of the bag and gave it to Jack. She reached into the bag again and took   | out the apple.  ✅  | out the carrot. ✅
Alice and Jack walked up the street and met a girl in a red dress. The girl said to them, "Hi, I am Jane. What are your names?"   | The girl smiled and said, "My name is Alice. Nice to meet you, Alice."  ❌  | Alice said, "I'm Alice and this is Jack." ✅
Diva was hungry, and wanted to bake a cake, but she did not have any sugar at home, so she decided to go ask around. She started walking and met a squirrel. She asked the squirrel, "Would you happen   | The squirrel nodded and said, "Yes, I would love to bake a cake." Diva and the squirrel gathered all the ingredients together and poured them into a bowl." ✅ | to have some sugar?" ✅
