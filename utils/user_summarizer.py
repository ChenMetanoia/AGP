from transformers import pipeline
from agentlite.llm.LLMConfig import LLMConfig
import torch


class UserSummarizer(BaseLLM):
    def __init__(self,):
        # Initialize the LLM with the desired configuration
        llm_config_dict = {"llm_name": "llama3", "temperature": 1, "api_key":"EMPTY", "base_url":"http://localhost:8000/v1/"}
        llm_config = LLMConfig(llm_config_dict)
        self.llm = pipeline(task='text-generation', 
                            model='meta-llama/Meta-Llama-3-8B-Instruct', 
                            device='cuda:0',
                            max_new_tokens=128,
                            return_full_text=False, # whether return full text
                            model_kwargs={"torch_dtype": torch.bfloat16})
        self.EOS = '|endoftext|'
        super().__init__(llm_config)
        
    def run(self, prompt: str):
        result = self.llm(prompt)
        generated_text = result[0]['generated_text']  # Corrected key from 'generated text' to 'generated_text'
        generated_text = generated_text.split(self.EOS)[0]
        return generated_text
    
    def write_user_summary(self, movie_profile_seq, summary_length=10):
        if len(movie_profile_seq) < summary_length:
            template = self.user_summary_template(movie_profile_seq)
            user_summary = self.run(template)
        else:
            for i in range(0, len(movie_profile_seq), summary_length):
                if i == 0:
                    template = self.user_summary_template(movie_profile_seq[i:i+summary_length])
                    user_summary = self.run(template)
                else:
                    template = self.user_summary_template(movie_profile_seq[i:i+summary_length], user_summary)
                    user_summary = self.run(template)
        return user_summary
        
    def user_summary_template(self, movie_profile_seq, previouse_summary=None):
        if previouse_summary is None:
            template = f"""
            [Instruction]
            Given the sequence of items interacted with by a user, ordered by timestamp, please provide a concise summary of the user's preferences based on observed trends in their interactions. This summary will be utilized to create a user summary for enhancing the recommendation system.
            
            [Example Start]
            [User Sequence]
            ['This adaptation of Charlotte Brontë\'s classic novel brings the timeless tale of Jane Eyre to life with a poignant and emotional portrayal. The film\'s atmospheric setting, coupled with Samantha Morton\'s captivating performance as the titular character, will resonate with fans of period dramas and literary adaptations. Viewers who appreciate character-driven stories, intricate plot development, and strong female protagonists will find this film to be a compelling and immersive experience. It is particularly suited for those who enjoy British period dramas, literary classics, and character-driven narratives with a focus on female empowerment.', 'This retro-style adventure series, starring Patrick Macnee, brings together a team of spies and agents to combat international threats. With its blend of espionage, action, and humor, it appeals to fans of classic TV shows and spy thrillers. Those who enjoy retro-themed entertainment, appreciate British humor, and are drawn to the 1960s spy genre will find this series captivating. It is particularly well-suited for viewers who enjoy nostalgia-driven content, fans of classic TV shows, and those seeking a light-hearted yet engaging viewing experience.', '    This documentary series explores the infamous Jack the Ripper case, offering a gripping and suspenseful journey through the streets of Victorian London. With its focus on historical true crime, it appeals to fans of mystery, crime, and investigative programming. Viewers who enjoy exploring the darker side of history, analyzing evidence, and piecing together the puzzles of the past will find this series captivating. Those with a keen interest in true crime stories, historical events, and forensic analysis will also appreciate the meticulous research and attention to detail presented in this documentary. It is particularly suited for enthusiasts of crime documentaries, historical mysteries, and those seeking a thought-provoking and thrilling viewing experience.```', "    This classic British period drama follows the lives of the aristocratic Bellamy family and their loyal servants as they navigate the complexities of their intertwined worlds. The show's intricate storytelling, memorable characters, and meticulous attention to historical detail will captivate viewers who appreciate period dramas, British television, and nuanced character development. Fans of shows like Downton Abbey and Victoria will find this series to be a compelling addition to their viewing schedule. It is particularly suited for those who enjoy historical fiction, character-driven storytelling, and the exploration of social class dynamics. The show's blend of romance, drama, and intrigue will also appeal to fans of mystery and suspense genres.```", '     This gripping drama, starring Jeff Daniels, delves into the lives of individuals affected by a tragic event. With its intense and emotional storytelling, it resonates with viewers who appreciate character-driven narratives and are drawn to stories of human struggle and resilience. Fans of dramatic series and those who enjoy character-centric storytelling will find this show to be a compelling watch. It is particularly suited for audiences who appreciate complex characters, nuanced plot development, and thought-provoking themes, making it an excellent choice for viewers seeking a captivating and emotionally charged viewing experience.']

            [User Summary]
            The user exhibits a strong affinity for character-rich, narrative-driven content, particularly within historical or period settings. They favor British period dramas and literary adaptations noted for their detailed storytelling and atmospheric depth. Additionally, they are drawn to classic spy series with a nostalgic touch and historical true crime documentaries that offer a deep, investigative approach into darker historical events. The preference extends to emotionally intense dramas that explore complex social dynamics and personal resilience. 
            {self.EOS}
            [Example End]

            [User Sequence]
            {movie_profile_seq}
            
            [User Summary]
            """
        else:
            template = f"""
            [Instruction]
            Review the forthcoming batch of items interacted with by a user, organized by timestamp, and consider the existing user summary provided. Produce a concise summary that captures the user's evolving preferences based on recent interactions. This updated summary should mirror the style of the existing user summary and will be used to refine and improve the user profile in our recommendation system.
            
            [Example Start]
            [User Sequence]
            ['This classic British television series follows the life of a brilliant and witty barrister, Rumpole of the Bailey, as he navigates the complexities of the legal system. With its blend of humor, wit, and clever plot twists, it appeals to viewers who appreciate clever storytelling, strong characters, and British charm. Ideal for fans of classic TV dramas, legal procedurals, and British culture, this series is especially attractive. Enthusiasts of character-driven storytelling, as well as individuals who appreciate clever dialogue and nuanced character development, will find this series to be a compelling addition to their viewing schedule. It is particularly suited for viewers who enjoy British period dramas, courtroom dramas, and character-driven stories.', '     This film, directed by Aleksandr Lazarev, is a poignant and thought-provoking exploration of the complexities of human relationships. With its focus on the intricate dynamics between individuals and the consequences of their actions, it resonates with viewers who appreciate character-driven stories and nuanced portrayals of human nature. Ideal for audiences who enjoy psychological dramas, character-driven narratives, and films that delve into the complexities of human relationships, this movie is particularly suited for fans of thought-provoking cinema and those seeking a more introspective viewing experience. It is also an excellent choice for viewers who appreciate international productions and are eager to discover new perspectives on universal themes.', '    This documentary-style film focuses on the art of competitive dance, providing insights into the strategies and techniques used by professional dancers to outperform their opponents. With its emphasis on skill, athleticism, and mental preparation, it appeals to viewers who appreciate the fusion of art and sport. Ideal for fans of dance and competition-based content, as well as individuals interested in personal development and self-improvement, this film is particularly suited for those who enjoy documentaries and sports-related programming. It is also a great choice for anyone seeking inspiration and motivation to push their own boundaries and strive for excellence. ', "     This classic episode of Star Trek - The Original Series, featuring Captain Kirk and his crew, takes viewers on an action-packed journey through the galaxy. With its blend of science fiction, adventure, and social commentary, it appeals to fans of the Star Trek franchise, science fiction enthusiasts, and those who enjoy classic television. Ideal for viewers who appreciate the original series' nostalgic value, its memorable characters, and the show's exploration of complex themes, this episode is especially attractive. Fans of Captain Kirk and the Enterprise crew, as well as individuals who appreciate well-crafted storytelling and memorable characters, will find this episode to be a compelling addition to their viewing schedule. It is particularly suited for those who enjoy classic science fiction, retro television, and the Star Trek franchise."]
            
            [Existing User Summary]
            The user exhibits a strong affinity for character-rich, narrative-driven content, particularly within historical or period settings. They favor British period dramas and literary adaptations noted for their detailed storytelling and atmospheric depth. Additionally, they are drawn to classic spy series with a nostalgic touch and historical true crime documentaries that offer a deep, investigative approach into darker historical events. The preference extends to emotionally intense dramas that explore complex social dynamics and personal resilience. 
            
            [User Summary]
            The user maintains a deep appreciation for character-driven, narrative-rich content, especially within historical or period contexts and clever British productions. Their recent interactions extend this preference to include psychological dramas that delve into human relationships, competitive dance documentaries highlighting skill and strategy, and classic science fiction series like "Star Trek." These selections emphasize a continued affinity for content that combines intellectual stimulation with emotional engagement, featuring complex character development and intricate storytelling. This blend of interests positions the user well for recommendations that are thought-provoking and character-focused, spanning both classic and contemporary settings.
            {self.EOS}
            [Example End]

            [User Sequence]
            {movie_profile_seq}
            
            [Existing User Summary]
            {previouse_summary}
            
            [User Summary]
            """
            
        return template
