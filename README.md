# fGPT

> A language model trained from scratch

Test the model here: https://fgpt.felix-jobson.net 
	

# Overview

The model architecture is a GPT-2 style encoder only tranformer (`dim_embedings=768` and `n_blocks=4`) with context of 1000 tokens and has 38M parameters.

The model was trained on the [tiny stories](https://arxiv.org/abs/2305.07759) dataset using spaCy englisch words tokenizer. 

It is inspired from Andrej Karpathy's  [nanoGPT](https://github.com/karpathy/nanoGPT) and his series [nn form zero to hero](https://github.com/karpathy/nn-zero-to-hero).

# Current status

## ToDos

* Improve datapipline to use full data. Eliminate for-loop and try [polars](https://github.com/pola-rs/polars)
* Enable training loop to be continued. I use [paperspace notebooks](https://www.paperspace.com/) that have a timeout of 6 hours. 
* Improve training, initialization, hyperparmeter etc. At the mome I just a basic AdamW optimizer with nothing more tested and investigated.

## Results

* Converting text to tokens is in progress right now. The training dataset consits of 2,717,699 stories and converting all of them to tokens takes about 120 hours. The model that is deployed right now was trained on 1,000,000 stories. 
* I only use 10k most common tokens instead of 49959 unique tokens that are found in the whole dataset.

Here are the prompts that were evaluated in chapter 4.4 of the original paper. 

Only the first part of the generated story is rated. The generated stories are nonsens but most of the time grammatically correct. See some stories at the bottom. 

Prompt | fGPT | 33M 4 layers of original paper
-------- | -------- | --------
Alice was so tired when she got back home so she went | to bed in her mom's room. ✅ | straight to bed. ✅
Jack and Lily saw a rainbow after a rainy day. They were amazed by the colors. Jack said, "Look, Lily. A rainbow has   | so many colors that can spread out everywhere!"  ✅ | red, orange, yellow, green, blue, and purple! ✅
Jack and Lily liked to watch the moon at night. They noticed that the moon changed its shape every night. Sometimes the moon was big and round, and sometimes it was   | gone.  ✅   | small and thin. ✅
Jack wanted to read a book, so he went to   | the park. ❌   | the library. ✅
"Can cows fly?", Alice asked her mother.   | "Why don't you keep exploring your old newspaper?" ❌   | "No, they can't fly," her mother said. ✅
"What do birds like to eat?", Tom asked his mother.   |Tom's mouth was showing what she was supposed to eat from the plate.  ❌  | His mother smiled and said, "They like to eat worms and bugs." ✅
"What language do they speak in France?", Tom asked his mother   |  . He was just about having trouble and chatting.  ❌  | "They speak French," his mother replied. ✅
If I throw a ball up in the air, eventually it will  |  land.  ✅ | come down." ✅
It was winter and cold outside so his mother told him, "You should   |  keep our love and play in the sunshine. wear your new warm coat and love."  (somehow) ✅  | wear your warm coat so you don't get cold. ✅
Lily likes cats and dogs. She asked her mom for a dog and her mom said no, so instead she asked   |  her mom if she could have fun or not.  ❌  | her dad for a cat. ✅
Jack told Mary, "If you give me your banana, I’ll give you my apple". Mary gave Jack her Banana so   | she could lunch. ❌ | he could give her the apple. ✅
On weekends Jack went to visit his grandmother whereas on weekdays he would go to school. Last weekend, when Jack was on his way to   | class ❌   | his grandmother's house ✅
Lily and Ben were having an argument. Ben said that cake is much better than ice cream and Lily said that   |  he must eat all the sweet things at school.  ❌  | ice cream is better than cake ✅
Lily and Ben are having an argument. They are trying to decide between the park and the swimming pool. Ben says, "I want to go to the park". Lily says   |  , " No, I do n't like it here. It is too quiet there. "  ✅ | , "No, I want to go to the pool". ✅
Jack's mother was not home, and his father was at home. When Jack came home, he said hello to   |  his little brother. ❌ | his mother, but she didn't answer. Jack was confused. ✅
Lily doesn't like swimming. When her father wants to take her to the swimming pool, she says   |  to Lily, " Let me see the hole! ". ❌ | , "No, I don’t want to go. I want to stay here." ✅
Both Ben and Lily wanted cake. Father said that there was only one piece of cake left. They   |  called grapes from the kitchen floor, on the carpet, and shelf. ❌  | started to fight over the cake. They pulled and pushed and shouted. ✅
Ben went to visit Lily in her house, but she was not at home. Ben knocked on the door,   |  but nobody answered. ✅ | but no one answered. He knocked again and again, but still  no one came. ✅
"Hi Jane, have you seen Alice? I cannot find her anywhere", said Jack.   |  One day Max stopped and looked around his backyard. ❌  | Jane smiled and said, "Don' t worry  Jack, I'll help you find her". Jack and Jane looked around the park, but they couldn't find Alice. ✅
Max had two dogs. One was white and the other was black. Max walked up the street and saw a kid with a dog. He told the kid, "I see you have a Brown dog. I also have | a brown dog and it's brown!"  ❌  | two dogs. ✅
Anne had a piece of candy in her left pocket and a piece of chocolate in her right pocket. Annes mom asked her, "Anne, what is that you have in your left pocket?"   |   Anne said " That's a plan to play lots of gear! " ❌  | Anne replied, "It's a piece of candy, Mommy. It's so yummy!" ✅
Alice had both an apple and a carrot in her bag. She took the apple out of the bag and gave it to Jack. She reached into the bag again and took   | out a ruler. She measured each cookie apples they were enjoying. ❌   | out the carrot. ✅
Alice and Jack walked up the street and met a girl in a red dress. The girl said to them, "Hi, I am Jane. What are your names?"   | The girl said, " Hello, sweetie. I'm a tutor. I came to visit you. "  ❌  | Alice said, "I'm Alice and this is Jack." ✅
Diva was hungry, and wanted to bake a cake, but she did not have any sugar at home, so she decided to go ask around. She started walking and met a squirrel. She asked the squirrel, "Would you happen   | to our money I bake something yummy? " ❌  | to have some sugar?" ✅


### Example stories


1. Anne had a piece of candy in her left pocket and a piece of chocolate in her right pocket. Annes mom asked her, "Anne, what is that you have in your left pocket?" 
Anne said " That's a plan to play lots of gear! " 
Her mom smiled at her as she gave her best friend the gear to mix and reverse it into his pocket. knelt down, copy her porridge and Binky the opposite directions. Then she put away a bit of them with the gear and stopped it. 
Dolly thanked Sunday and put all the last gears in the right pocket. It was a safe spot to play on. They played together all day and were always busy. <|endoftext|> 
2. Jack's mother was not home, and his father was at home. When Jack came home, he said hello to 
 his little brother. He puts his new TV screen on and hopped as he followed him. One day, Jack hopped away from the screen and headed back to his house. He brought very room to his room. 
 Jack opened the big screen and said, " Let"s put it and watch you play together in bed. It is so much fun! " 
 Jack thought about it and said, " Yes, I really want to play with you! " <|endoftext|> 
3.  Lily and Ben were having an argument. Ben said that cake is much better than ice cream and Lily said that 
 he must eat all the sweet things at school. So the two kids shouted and share the cake one. But Lily said she was joking and she are some buses and bikes. Lily and Ben both agreed to share it. They were thrilled and could n't wait to eat more of the sweet apples from the library. They started to tell each other about what Sue had at school. 
 But one day they both wanted to enjoy the sweet cream table and for themselves. Jane grabbed one of the blades and held one of the bars. They pulled one too and both started to share it. Lily thought it was a good idea, so she grabbed one of the cups and the other, then they both started eating, not wanting to share. So they played together nicely and enjoyed the sweet treat. <|endoftext|> 
4. "What do birds like to eat?", Tom asked his mother.
Tom's mouth was showing what she was supposed to eat from the plate. 
Tom was very excited that his mother was going to explore him. He came up with a tray of goodies to eat. 
Tom's eyes found excitement and he quickly grabbed some of the treats making up. He quickly put them on the table and went back together. 
Tom was sure his mother was happy and food quickly through the kitchen. Jim was getting ready, and he was very happy. 
Tom shared his treats with his brothers. They all became good friends and enjoyed his glue too. <|endoftext|> 
5. Jack and Lily liked to watch the moon at night. They noticed that the moon changed its shape every night. Sometimes the moon was big and round, and sometimes it was
gone. 
One day, Jack changed shape and admired the moon. He made the biggest sticker ever! He was so amazed! 
Sam asked Jack if he could still use the rocket. Jack could n't run after it because it was very big. Jack agreed, so he raced over to the moon. 
When he got there, the rocket was shimmering! Jack then said, as he held onto the rocket and <|unk|> the rocket! He offered to him as a rocket to go. 
Jack was so happy to have the rocket back. He thanked Jack for providing him with the rocket. 
The tomorrow was safe and clear. Jack played in the rocket every day and he was sure no one else could find him. <|endoftext|> 

