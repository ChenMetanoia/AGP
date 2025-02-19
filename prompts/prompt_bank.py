import sys

class PromptBank:
    def __init__(self, BOS, EOS):
        self.BOS = BOS
        self.EOS = EOS
        
        self.STARTING_ITEM_PROFILE_GENERATION_PROMPT = """
        Based on the current user_profile and the set of item_profiles that the user has interacted with, generate a new item_profile that is most likely to align with the user’s next potential interest. The generated item_profile should follow the same format as the previously interacted item profiles and should include the following considerations:

        1.	Attribute Consistency: Ensure that the new item profile maintains consistency with the attributes the user has shown interest in, as reflected in the user_profile and previous item_profiles.
        2.	Diversity and Novelty: While aligning with the user’s known preferences, introduce slight variations or new attributes to encourage exploration of potentially new interests.
        3.	Preference Continuity: Balance the introduction of novel attributes with the continuity of attributes the user is consistently drawn to, ensuring a smooth transition between past interactions and future recommendations.
        4.	Engagement Potential: Prioritize attributes that have the highest likelihood of engaging the user based on their historical interaction patterns.
        
        The goal is to generate a new item_profile that accurately predicts and aligns with the user’s evolving interests, fostering both engagement and discovery.
        """

        self.STARTING_ITEM_PROFILE_GENERATION_PROMPT_SHORT = """
        Based on the current user_profile and the set of item_profiles that the user has interacted with, generate a new item_profile that is most likely to align with the user’s next potential interest, fostering both engagement and discovery.
        """
        
        self.STARTING_ITEM_PROFILE_GENERATION_PROMPT_RERANK = """
        First priority: Consider users' favorite movie genres.
        Second priority: Consider users' favoratie movie years.
        Third priority: Consider users' favoratie director and actors.
        """

        self.STARTING_USER_PROFILE_GENERATION_PROMPT = """
        """

        self.SYSTEM_PROMPT = """
        You are a recommender system. Your objective is to generate a synthetic item profile that can be encoded into an embedding using a pre-trained language model. This embedding will be used to calculate similarity and generate a recommendation ranking. To achieve this, create a learnable prompt directly. The prompt should guide the generation of an item profile output with no extraneous context. You have full autonomy over the design of this prompt. Ranking results will be provided to help you optimize and refine your learnable prompt.
        """

        
        self.SYSTEM_ROLE_DESCRIPTION = """
        Description:
        The Adaptive Recommender System Engine is designed to dynamically generate and update user and item profiles to enhance the relevance and personalization of recommendations. This system operates through two core functions:

            1.	User Profile Generation and Update:
            •	The system takes into account a user’s current profile along with their interactions with various items. Each interaction is represented by an item_profile that details the attributes of the item.
            •	Using this information, the system generates or updates the user_profile to reflect the user’s evolving preferences, interests, and tendencies.
            •	It identifies the relevance of item attributes to the user’s preferences, tracks the evolution of these preferences, and uncovers emerging trends to maintain an accurate and up-to-date user profile.
on for Recommendation:
            •	Based on the user_profile and the item_profiles that the user has interacted with, the system generates new item_profiles that are likely to capture the user’s next potential interests.
            •	The generated item profiles follow the format of previously interacted items, ensuring consistency while introducing diversity and novelty.
            •	The system aims to balance familiar attributes with new, engaging ones to encourage exploration while ensuring relevance to the user’s preferences.

        Objective:
        The ultimate goal of the Adaptive Recommender System Engine is to create a personalized and engaging recommendation experience by continuously refining user profiles and predicting items that align with the user’s current and emerging interests.
        """

        GLOSSARY_TEXT = """
        ### Glossary of tags that will be sent to you:
        # - <LM_SYSTEM_PROMPT>: The system prompt for the language model.
        # - <LM_INPUT>: The input to the language model.
        # - <LM_OUTPUT>: The output of the language model.
        # - <FEEDBACK>: The feedback to the variable.
        # - <CONVERSATION>: The conversation history.
        # - <FOCUS>: The focus of the optimization.
        # - <ROLE>: The role description of the variable."""
        
        self.OPTIMIZER_SYSTEM_PROMPT = (
        "You are part of an optimization system that improves reranking prompt as variable. "
        "Pay attention to priorityies of factors that affect user preferences in movies."
        "Pay attention to the role description of the variable, and the context in which it is used. "
        "You will receive some feedback, and use the feedback to improve the variable. "
        "This is very important: You MUST give your response by sending the improved variable between {new_variable_start_tag} {{improved variable}} {new_variable_end_tag} tags. "
        "The text you send between the tags will directly replace the variable.\n\n"
        f"{GLOSSARY_TEXT}"
    )

    def inject_user_profile_generation_prompt(self, user_profile_generation_prompt: str, user_profile: str, interaction: str) -> str:
        prompt = f"""
        [Instruction]
        {user_profile_generation_prompt}
        The output shoulde be a updated user profile based on the user's interaction with the item profile. You must generate user profile only.
        
        [Current User Profile]
        {user_profile}
        
        [Interaction]
        {interaction}
        
        [Updated User Profile]
        """
        return prompt

    def inject_item_profile_generation_prompt(self, item_profile_generation_prompt: str, user_profile: str, item_profiles: str) -> str:
        prompt = f"""
        [Instruction]
        {item_profile_generation_prompt}
        You MUST generate a new item profile based on the user's preferences and interactions with the item profiles. The output MUST strictly follow the format provided in the [New Item Profile Example] and MUST start with {self.BOS}, end with {self.EOS}. NO Title, No bold format.

        [User Profile]
        {user_profile}

        [Item Profiles]
        {item_profiles}

        <Start Example Output>
        {self.BOS}
        Santa Fe Rules is a captivating and suspenseful novel that delves into the world of mystery and intrigue. This book, written by acclaimed author Stuart Woods, masterfully weaves together elements of crime, politics, and romance, making it a thrilling read for fans of the genre. Perfect for those who enjoy complex characters, intricate plots, and a touch of humor, Santa Fe Rules is an ideal choice for readers who appreciate a good mystery. It is especially suited for enthusiasts of crime fiction, fans of Stuart Woods, and individuals who enjoy a gripping and unpredictable storyline. Those seeking a captivating and engaging read will find Santa Fe. 
        {self.EOS}
        <End Example Output>
        """
        return prompt
    
    def inject_item_profile_generation_prompt_no_user(self, item_profile_generation_prompt: str, item_profiles: str) -> str:
        prompt = f"""
        Instruction]
        {item_profile_generation_prompt}
        You MUST generate a new item profile based on the user's preferences and interactions with the item profiles. The output MUST strictly follow the format provided in the [New Item Profile Example] and MUST start with {self.BOS}, end with {self.EOS}. NO Title, No bold format.
        
        [Item Profiles]
        {item_profiles}
        
        <Start Example Output>
        {self.BOS}
        Santa Fe Rules is a captivating and suspenseful novel that delves into the world of mystery and intrigue. This book, written by acclaimed author Stuart Woods, masterfully weaves together elements of crime, politics, and romance, making it a thrilling read for fans of the genre. Perfect for those who enjoy complex characters, intricate plots, and a touch of humor, Santa Fe Rules is an ideal choice for readers who appreciate a good mystery. It is especially suited for enthusiasts of crime fiction, fans of Stuart Woods, and individuals who enjoy a gripping and unpredictable storyline. Those seeking a captivating and engaging read will find Santa Fe. 
        {self.EOS}
        <End Example Output>
        """
        return prompt

    def inject_item_profile_generation_prompt_rerank(self, item_profile_generation_prompt:str, historical_item_profiles:str, cf_item_profiles:str, sr_item_profiles:str) -> str:
        predicted_item_list = cf_item_profiles + sr_item_profiles
        seq_length = len(predicted_item_list)
        prompt = f"""
        [Instruction]
        {item_profile_generation_prompt}
        You MUST rank the items in Pretrained Item Profile List according to the user's Historical Item Profiles. Each element in Pretrained Item Profile is in the format of <item id: item profile>. The output MUST be a ranking list of Item IDs as integers from 0 to {seq_length-1} with length of {seq_length}. A smaller number means a higher rank.

        [Historical Item Profiles]
        {historical_item_profiles}

        [Pretrained Item Profile List]
        {predicted_item_list}
        
        """
        return prompt
    
    def inject_item_profile_generation_prompt_recall_one(self, item_profile_generation_prompt:str, historical_item_profiles:str, cf_item_profiles:str, sr_item_profiles:str) -> str:
        predicted_item_list = cf_item_profiles + sr_item_profiles
        seq_length = len(predicted_item_list)
        prompt = f"""
        [Instruction]
        {item_profile_generation_prompt}
        You MUST find the target item from Pretrained Item Profile List according to the user's Historical Item Profiles. Each element in Pretrained Item Profile is in the format of <item id: item profile>. The output MUST be a ranking list of Item IDs as integers from 0 to {seq_length-1} with length of {seq_length}, where the predicted target item id is the first integer in the list.
        
        [Historical Item Profiles]
        {historical_item_profiles}

        [Pretrained Item Profile List]
        {predicted_item_list}
        
        """
        return prompt
    
    def create_user_profile_prompt(self, user_profile_generation_instruction: str, historical_item_profiles: str) -> str:
        """
        Generate a prompt for creating a user profile based on historical item profiles.
        
        Args:
            user_profile_generation_instruction (str): Instructions for generating the user profile.
            historical_item_profiles (str): A detailed description of the user's historical item interactions.
        
        Returns:
            str: A formatted prompt to guide the generation of a user profile.
        """
        prompt = f"""
        [Task]
        Your goal is to generate a comprehensive user profile based on the user's historical item interactions.
        This user profile will describe the user's preferences and provide information to assist in reranking recommendations.

        [Instructions]
        {user_profile_generation_instruction}
        Use the historical item profiles provided below as the basis for creating the user profile. The output is the user profile ONLY.

        [Historical Item Profiles]
        {historical_item_profiles}
        """
        return prompt

    def rerank_by_user_profile_prompt(self, user_profile:str, sr_item_profiles:str, rerank_len:int) -> str:
        prompt = f"""
        [Instruction]
        Rerank item id in Pretrained Item Profile List according to User Profile. Output MUST be a list of integers from 0 to {rerank_len-1} separated by commas, with length of {rerank_len}, where the predicted target item id is the first integer in the list. Output MUST start with {self.BOS}, end with {self.EOS}.

        [User Profile]
        {user_profile}

        [Pretrained Item Profile List in format of <item id: item profile>] 
        {sr_item_profiles}
        """
        return prompt

    def inject_item_profile_generation_prompt_recall_single(self, item_profile_generation_prompt:str, historical_item_profiles:str, sr_item_profiles:str, rerank_len:int) -> str:
        prompt = f"""
        [Instruction]
        {item_profile_generation_prompt}
        Output MUST be a list of integers from 0 to {rerank_len-1} separated by commas, with length of {rerank_len}, where the predicted target item id is the first integer in the list. Output MUST start with {self.BOS}, end with {self.EOS}.

        [Historical Item Profiles]
        {historical_item_profiles}

        [Pretrained Item Profile List in format of <item id: item profile>] 
        {sr_item_profiles}
        
        """
        return prompt

    def user_profile_evaluation_reason(self, ground_truth_item_profile: str, reranked_item_profiles: str, 
                                    generated_user_profile: str, user_profile_prompt: str) -> str:
        """
        Create a prompt to evaluate the generated user profile and provide actionable instructions for optimizing the prompt.
        
        Args:
            ground_truth_item_profile (str): The profile of the ground-truth item reflecting the user's actual preference.
            reranked_item_profiles (str): The current list of reranked item profiles.
            generated_user_profile (str): The user profile generated by the system.
            user_profile_prompt (str): The prompt used for generating the user profile.
        
        Returns:
            str: A formatted prompt to guide the evaluation and generate optimization instructions.
        """
        prompt = f"""
        [Task]
        Evaluate the generated user profile and provide actionable instructions to optimize the user profile generation prompt.
        The goal is to improve alignment between the Ground-truth Item and the predicted reranked results.

        [Instructions]
        1. Analyze the Ground-truth Item Profile and Current Reranked Item Profile List:
        - Identify why the Ground-truth Item is not ranked first (if applicable).
        - Highlight any alignment or misalignment between the Ground-truth Item and the Generated User Profile.
        
        2. Review the Generated User Profile:
        - Identify features that correctly represent the user's preferences.
        - Pinpoint any missing or irrelevant features that may have caused misalignment.
        
        3. Review the Current User Generation Prompt:
        - Suggest modifications to better capture the user's preferences for future optimization.
        - Retain any useful elements that effectively represent user preferences.

        4. Provide actionable instructions for optimizing the user profile generation prompt:
        - Specify changes to improve the Generated User Profile.
        - Explain how these changes can enhance alignment with the Ground-truth Item.

        [Ground-truth Item Profile]
        {ground_truth_item_profile}

        [Current Reranked Item Profile List]
        {reranked_item_profiles}

        [Generated User Profile]
        {generated_user_profile}

        [Current User Generation Prompt]
        {user_profile_prompt}
        
        [Expected Output]
        Provide actionable instructions for optimizing the user profile generation prompt with the following details:
        1. Identify the key issues with the Generated User Profile based on the Ground-truth Item and reranked results.
        2. Propose specific changes to the User Generation Prompt to improve alignment with the user's preferences.
        3. Highlight features of the current prompt and user profile that should be retained for future optimization.
        """
        return prompt

    def reranking_evaluation_reason(self, predicted_ranking_list, y_item_id, y_item_profile, sr_item_profiles ,historical_item_profiles):
        prompt = f"""
        [Instruction]
        You are given a task to evaluate a Predicted Item List reranked for a user, accordding to the Ground-truth Item Id which should be the first in the list. 
        Compare the Ground-truth Item Profile, All Item Profiles in Predicted Item List, and the Historical Item Profiles of those items interacted by the user.
        Your output MUST use 3 sentences to explain (1) why the Ground-truth Item is not the first one in the Predicted Item List, and (2) why user likes the Ground-truth Item.

        [Predicted Item List]
        {predicted_ranking_list}

        [Ground-truth Item Id]
        {y_item_id}

        [Ground-truth Item Profile]
        {y_item_profile}

        [All Item Profiles in format of <item id: item profile>]
        {sr_item_profiles}
        
        [Historical Item Profiles]
        {historical_item_profiles}

        """
        return prompt
    
    def generate_optimization_prompt(self, original_rerank_prompt: str, actionable_instructions: list, 
                                     distance: int = None, generalized_failed_cases: str = None) -> str:
        """
        Generate a prompt to optimize the user profile generation prompt based on actionable instructions.
        
        Args:
            original_rerank_prompt (str): The original prompt used for reranking the system.
            actionable_instructions (list): A list of actionable instructions for future optimization.
            distance (int, optional): Update strength (range 1–10) to indicate how much the new prompt 
                                    should differ from the original prompt. Default is None.
        
        Returns:
            str: A formatted prompt for optimizing the rerank prompt.
        """
        # Count total cases and failed cases
        batch_size = len(actionable_instructions)
        filtered_instructions = [
            f"Instruction {i}: {instruction}" for i, instruction in enumerate(actionable_instructions) if instruction!=None
        ]
        if generalized_failed_cases is None:
            failed_cases = '\n'.join(filtered_instructions)
        else:
            failed_cases = generalized_failed_cases
        incorrect_num = len(filtered_instructions)
        # Base instruction
        instruction = f"""
        [Instruction]
        The previous prompt achieved an error rate of {incorrect_num} out of {batch_size} and needs improvement
        based on actionable instructions generated from the evaluation process.
        Your task is to optimize the prompt by incorporating the lessons from the instructions listed below.
        Your output MUST only include the optimized prompt, formatted as follows:
        - Begin with {self.BOS}
        - End with {self.EOS}
        
        Your output will replace the current prompt for the rerank system.
        """
        
        # Add optional update strength information
        if distance is not None:
            instruction += f"""
        The 'Update Strength' ranges from 1 to 10 and indicates how much the new prompt should differ from the original one.
        Higher values suggest a more significant modification, while lower values indicate minor adjustments.
        """

        # Combine components
        prompt = f"""
        {instruction}
        
        [Previous Rerank Prompt]
        {original_rerank_prompt}

        [Actionable Instructions for Optimization]
        {failed_cases}
        """
        
        if distance is not None:
            prompt += f"\n[Update Strength]\n{distance}"
        
        return prompt

    def generalize_failed_cases_prompt(self, failed_cases: list) -> str:
        """
        Generate a generalized instruction prompt to optimize the user profile generation prompt.
    
        Args:
            failed_cases (list): A list of specific failed case analyses (strings).
    
        Returns:
            str: A generalized instruction prompt with actionable guidelines for optimization.
        """
        # Generalized instruction to address specific reasons in failed cases
        general_instruction = (
            "[Generalized Instruction]\n"
            "To improve the user profile generation prompt, focus on creating instructions that are flexible and adaptive to various scenarios. "
            "The optimization should aim to enhance the system's ability to generalize across unseen users and contexts, addressing potential limitations such as incomplete data, ambiguous preferences, "
            "or diversity in user behaviors. Avoid overfitting to specific cases and ensure the instructions emphasize robustness, clarity, and adaptability.\n"
            "Below are specific failed cases to inform the optimization process. Use these as examples to derive broader patterns for improving the prompt."
        )
    
        # Combine general instruction with failed cases
        failed_cases_text = "\n".join(failed_cases)
    
        return f"{general_instruction}\n\n[Failed Cases]\n{failed_cases_text}"

    def rerank_by_cot_prompt(self, historical_item_profiles: str, sr_item_profiles: str, rerank_len: int) -> str:
        prompt = f"""
        [Instruction]
        Think step-by-step based on the user's historical item interactions to determine item preferences. Then, rerank the given Pretrained Item Profile List accordingly. Output MUST be a list of integers from 0 to {rerank_len-1} separated by commas, with length of {rerank_len}. The first integer in the list represents the highest-ranked item. The output MUST start with {self.BOS}, end with {self.EOS}.
    
        [User Historical Item Profiles]
        {historical_item_profiles}
    
        [Pretrained Item Profile List in format of <item id: item profile>] 
        {sr_item_profiles}
    
        [Step-by-Step Rationale]
        - Identify user preferences from historical interactions.
        - Compare these preferences with the given item profiles.
        - Rank items accordingly.
        - Ensure the output strictly follows the required format.
        
        """
        return prompt
    
    
    def rerank_directly_prompt(self, historical_item_profiles: str, sr_item_profiles: str, rerank_len: int) -> str:
        prompt = f"""
        [Instruction]
        Rerank item id in Pretrained Item Profile List based on User's Historical Item Profiles. Output MUST be a list of integers from 0 to {rerank_len-1} separated by commas, with length of {rerank_len}. The first integer in the list represents the highest-ranked item. The output MUST start with {self.BOS}, end with {self.EOS}.
    
        [User Historical Item Profiles]
        {historical_item_profiles}
    
        [Pretrained Item Profile List in format of <item id: item profile>] 
        {sr_item_profiles}
    
        """
        return prompt
