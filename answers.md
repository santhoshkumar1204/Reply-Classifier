# Part C – Short Answer (Reasoning)

#1. If you only had 200 labeled replies, how would you improve the model without collecting thousands more?
  
With only 200 labeled replies, I would rely on data augmentation techniques like paraphrasing, back-translation, or synonym replacement to increase the effective dataset size. Leveraging pre-trained transformer models like DistilBERT allows the model to benefit from prior language understanding even with few labeled examples. I would also use cross-validation and regularization to prevent overfitting and ensure the model generalizes well.

# 2. How would you ensure your reply classifier doesn’t produce biased or unsafe outputs in production? 
 
To prevent bias or unsafe outputs, I would test the model on diverse datasets representing different demographics and contexts. Monitoring model predictions continuously and adding human in the loop validation for edge cases would help catch unwanted behavior. Additionally, retraining the model periodically with updated and balanced data ensures fairness, reliability, and safety in production.

# 3. Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non-generic?  

I would include as much context as possible in the prompt, such as the recipient’s name, company, role, and past interactions. Providing examples of well-written openers and clear instructions helps the model generate specific and relevant content. Iteratively refining prompts based on output quality and testing multiple variations ensures the results are personalized, engaging as well as non repetitive.
