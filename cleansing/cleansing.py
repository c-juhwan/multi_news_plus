import os
import json
import time
import copy
import argparse
import multiprocessing as mp
from multiprocessing import Pool
import openai
from tqdm.auto import tqdm
openai.api_key = os.environ["OPENAI_API_KEY_ACL"]

def main(args: argparse.Namespace) -> None:
    with open(f"./{args.input_file_name}.json", "r") as f:
        raw_data = json.load(f)
    raw_data = raw_data[args.start_idx:args.end_idx]
    cleansed_data = copy.deepcopy(raw_data)
    print(f"Loaded file from {args.start_idx} to {args.end_idx} from {args.input_file_name}.json")

    for i in tqdm(range(0, len(raw_data))):
        result_count, reasons = get_multiple_agent_result(args, raw_data[i])

        # Exclude the document that is not relevant to the summary
        filter_key = [key for key, value in result_count.items() if value > args.num_agents / 2]

        cleansed_data[i]["cleansed_document"] = [item for idx, item in enumerate(cleansed_data[i]["document"]) if idx not in filter_key]
        cleansed_data[i]["raw_document"] = raw_data[i]["document"]
        cleansed_data[i]["document_num"] = len(cleansed_data[i]["cleansed_document"])
        cleansed_data[i]["reasons"] = reasons
        del cleansed_data[i]["document"]

        cleansed_data[i]["vote_count"] = result_count

        # time.sleep(2) # Sleep for 2 seconds to avoid the rate limit

    if not os.path.exists("./cleansed_data/"):
        # make directory if not exists
        os.makedirs("./cleansed_data/")

    with open(f"./cleansed_data/{args.output_file_name}_{args.cleansing_mode}_{args.start_idx}_{args.end_idx}.json", "w") as f:
        json.dump(cleansed_data, f, indent=4, ensure_ascii=False)

    print(f"Saved file to {args.output_file_name}_{args.cleansing_mode}_{args.start_idx}_{args.end_idx}.json")
    time.sleep(3) # Sleep for 3 seconds to avoid the rate limit

def get_multiple_agent_result(args: argparse.Namespace, raw_doc: dict) -> dict:
    # Call multiprocessing
    starmap_items = [
        (args, raw_doc) for _ in range(args.num_agents)
    ]

    with Pool(args.num_agents) as p:
        each_agents_result = p.starmap(get_agent_result, starmap_items)

    # Count the number of each result
    result_count = {}
    for i in range(0, raw_doc["document_num"]):
        result_count[i] = 0

    for i in range(0, args.num_agents):
        for each_value in each_agents_result[i][0]:
            if each_value != -1 and each_value < raw_doc["document_num"]:
                result_count[each_value-1] += 1 # -1 because the document number starts from 1
            else:
                continue

    reasons = [x[1] for x in each_agents_result]

    return result_count, reasons

def get_agent_result(args: argparse.Namespace, raw_doc: dict) -> list:
    raw_summary = raw_doc['summary']
    raw_documents = raw_doc['document']

    if args.cleansing_mode == 'zs':
        message_list = get_zs_message()
    elif args.cleansing_mode == 'shot':
        message_list = get_shot_message()
    elif args.cleansing_mode == 'cot':
        message_list = get_cot_message()

    content = f"[Summary]\n{raw_summary}\n"
    for i in range(len(raw_documents)):
        content += f"\n[Document {i+1}]\n{raw_documents[i]}"

    message_list.append({
        "role": "user",
        "content": content
    })

    error_counter = 0
    while True:
        try:
            response, message_list = get_response(args, message_list, attach=False)

            if args.cleansing_mode in ['zs', 'shot']:
                if "None" in response:
                    final_response = [-1]
                else:
                    if "|" in response:
                        final_response = response.split("|")
                        final_response = [int(x.split("Document ")[-1]) for x in final_response]
                    else:
                        final_response = [int(response.split("Document ")[-1])]
            elif args.cleansing_mode == 'cot':
                final_response = response.split("Therefore, ")[-1]
                # print(final_response, type(final_response))
                if "all" in [x.lower() for x in final_response.split()]:
                    final_response = [-1]
                elif "no" in [x.lower() for x in final_response.split()]:
                    final_response = [-1]
                else:
                    final_response = final_response.split("the irrelevant document is: ")[-1]
                    if "None" in final_response:
                        final_response = [-1]
                    elif "|" in final_response:
                        final_response = final_response.split("|")
                        final_response = [x.replace(".", "") for x in final_response]
                        final_response = [int(x.split("Document ")[-1]) for x in final_response]
                    else:
                        final_response = final_response.replace(".", "")
                        final_response = [int(final_response.split("Document ")[-1])]
            break # break the loop if no error
        except ValueError as e:
            # if the error message contains one integer value, it contains the document number
            # parse the document number and pass it to the final_response
            try:
                final_response = [int(e.args[0].split(" ")[0])] # The first integer value
                break
            except Exception as e:
                error_counter += 1
                time.sleep(1)
                if error_counter > args.error_patience:
                    final_response = [-1]
                else:
                    continue
        except openai.error.RateLimitError as e:
            print("RateLimitError: ", type(e).__name__, e)
            # parse the message
            required_time = e.args[0].split("Please try again in ")[-1]
            required_time = int(required_time.split("ms.")[0])
            required_time = required_time / 1000 # convert to seconds
            time.sleep(required_time * 3) # sleep for 3 times longer than the required time
            continue
        except Exception as e:
            print("Error: ", type(e).__name__, e)
            error_counter += 1

            time.sleep(1)
            if error_counter > args.error_patience:
                final_response = [-1]
            else:
                continue

    return final_response, response

def get_response(args: argparse.Namespace, message_list: list, attach=True):
    """
    Simple wrapper to get response from gpt.
    """

    gpt_response = openai.ChatCompletion.create(
        model=args.model_version, messages=message_list,
        temperature=args.gpt_temperature, max_tokens=args.gpt_max_token, top_p=args.gpt_top_p,
        frequency_penalty=args.gpt_freq_penalty, presence_penalty=args.gpt_presence_penalty
    )

    response = gpt_response['choices'][0]['message']['content']

    if attach:
        message_list.append({
            "role": "assistant",
            "content": response
        })

    return response, message_list

def get_zs_message() -> list:
    message_list = [
        {"role": "system", "content": "You are a helpful assistant. \
Your job is to identify the document that is not relevant to the summary. \
You must not generate chat-style responses such as \"Sure!\", \"Here's the output\". \
Please response only with the document number or 'None'\
If multiple documents are irrelevant to the summary, please resonse in the form of 'Document 1|Document 3'.\n"
        },
        {"role": "user", "content": "Given the summary and documents, please identify the document that is not relevant to the summary such as advertisements. If every document is relevant to the summary, output 'None'.\n\
Input document will be given in next dialogue.\n"
        },
    ]

    return message_list

def get_shot_message() -> list:
    message_list = [
        {"role": "system", "content": "You are a helpful assistant. \
Your job is to identify the document that is not relevant to the summary. \
You must not generate chat-style responses such as \"Sure!\", \"Here's the output\". \
Please response only with the document number or 'None'.\
If multiple documents are irrelevant to the summary, please resonse in the form of 'Document 1|Document 3'.\n"
        },
        {"role": "user", "content": "Given the summary and documents, please identify the document that is not relevant to the summary such as advertisements. If every document is relevant to the summary, output 'None'\n\
Input document will be given in next dialogue.\n"
        },
        {"role": "user", "content": """[Summary]\nThe unemployment rate dropped to 8.2 % last month, but the economy only added 120,000 jobs, when 203,000 new jobs had been predicted, according to today\'s jobs report. Reaction on the wall street journal\'s marketbeat blog was swift: " woah!!! bad number. " the unemployment rate, however, is better news; it had been expected to hold steady at 8.3 %. But the ap notes that the dip is mostly due to more americans giving up on seeking employment.\n\n[Document 1]\nNational archives yes, it\'s that time again, folks. It\'s the first friday of the month, when for one ever-so-brief moment the interests of wall street, washington and main street are all aligned on one thing: jobs. A fresh update on the u.s. Employment situation for january hits the wires at 8: 30 a.m. New york time offering one of the most important snapshots on how the economy fared during the previous month. Expectations are for 203,000 new jobs to be created, according to economists polled by dow jones newswires, compared to 227,000 jobs added in february. The unemployment rate is expected to hold steady at 8.3 %. Here at marketbeat hq, we \' ll be offering color commentary before and after the data crosses the wires. Feel free to weigh-in yourself, via the comments section. And while you\'re here, why don\'t you sign up to. Enjoy the show.\n[Document 2]\nEmployers pulled back sharply on hiring last month, a reminder that the u.s. Economy may not be growing fast enough to sustain robust job growth. The unemployment rate dipped, but mostly because more americans stopped looking for work. The labor department says the economy added 120,000 jobs in march, down from more than 200,000 in each of the previous three months. The unemployment rate fell to 8.2 percent, the lowest since january 2009. The rate dropped because fewer people searched for jobs. The official unemployment tally only includes those seeking work. The economy has added 858,000 jobs since december _ the best four months of hiring in two years. But federal reserve chairman ben bernanke has cautioned that the current hiring pace is unlikely to continue without more consumer spending.""",
        },
        {"role": "assistant", "content": "All"
        },
        {"role": "user", "content": """[Summary]\nShelly sterling plans " eventually " to divorce her estranged husband donald, she tells barbara walters at abc news. As for her stake in the los angeles clippers, she plans to keep it, the ap notes. Sterling says she would " absolutely " fight any nba decision to force her to sell the team. The team is her " legacy " to her family, she says. " to be honest with you, i \' m wondering if a wife of one of the owners … said those racial slurs, would they oust the husband? or would they leave the husband in? "\n\n[Document 1]\nLos angeles ( ap ) — in her first interview since the nba banned her estranged husband, shelly sterling says she will fight to keep her share of the los angeles clippers and plans one day to divorce donald sterling. ( click prev or next to continue viewing images. ) advertisement ( click prev or next to continue viewing images. ) los angeles clippers co-owner shelly sterling, below, watches the clippers play the oklahoma city thunder along with her attorney, pierce o\'donnell, in the first half of game 3 of the western conference... ( associated press ) shelly sterling spoke to barbara walters, and abc news posted a short story with excerpts from the conversation sunday. Nba commissioner adam silver has banned donald sterling for making racist comments and urged owners to force sterling to sell the team. Silver added that no decisions had been made about the rest of sterling\'s family. According to abc\'s story, shelly sterling told walters: " i will fight that decision. " sterling also said that she " eventually " will divorce her husband, and that she hadn\'t yet done so due to financial considerations.\n[Document 2]\nShelly sterling said today that " eventually, i am going to " divorce her estranged husband, donald sterling, and if the nba tries to force her to sell her half of the los angeles clippers, she would " absolutely " fight to keep her stake in the team. " i will fight that decision, " she told abc news \' barbara walters today in an exclusive interview. " to be honest with you, i \' m wondering if a wife of one of the owners, and there\'s 30 owners, did something like that, said those racial slurs, would they oust the husband? or would they leave the husband in? " sterling added that the clippers franchise is her " passion " and " legacy to my family. " " i\'ve been with the team for 33 years, through the good times and the bad times, " she added. These comments come nearly two weeks after nba commissioner adam silver announced a lifetime ban and a $ 2.5 million fine for donald sterling on april 29, following racist comments from the 80-year-old, which were caught on tape and released to the media. Read: barbara walters \' exclusive interview with v. Stiviano being estranged from her husband, shelly sterling said she would " have to accept " whatever punishment the nba handed down to him, but that her stake in the team should be separate. " i was shocked by what he said. And -- well, i guess whatever their decision is -- we have to live'""",
        },
        {"role": "assistant", "content": "All"
        },
        {"role": "user", "content": """[Summary]\nTucker carlson is in deep doodoo with conservative women after an ill-advised tweet referencing sarah palin that he posted, then removed, monday night. " palin\'s popularity falling in iowa, but maintains lead to become supreme commander of milfistan, " he tweeted — and we probably don\'t need to tell you where that is. His first attempt at an apology, which he tweeted the next morning: " apparently charlie sheen got control of my twitter account last night while i was at dinner. Apologies for his behavior. " that wasn\'t good enough for many conservative women, politico notes, rounding up reactions from bloggers to michelle malkin calling his behavior sexist and misogynistic. By late tuesday, carlson had offered up a more sincere-sounding apology: "i \' m sorry for last night\'s tweet. I meant absolutely no offense. Not the first dumb thing i\'ve said. Hopefully the last. " but at least one man — erick erickson, editor of redstate.com — was on carlson\'s side, tweeting his reaction to the post in question: " i laughed then got out my passport. "\n\n[Document 1]\nTucker carlson exposes his own sexism on twitter ( updated ) tucker carlson has done some good work in the past … his site, the daily caller, is a frequent stop of mine and many other conservatives. They were responsible for exposing the journolist scandal, which highlighted the planning and coordination of many members of the left-wing press. I will always be grateful to tucker\'s team for bringing that story to light. This is also why i am so angered by tucker\'s recent actions. I thought he was better than this. If you haven\'t heard by now, monday evening, tucker carlson posted a disturbing tweet about governor palin which said: palin\'s popularity falling in iowa, but maintains lead to become supreme commander of milfistan aside from tucker\'s sheep-like response to warped poll numbers, he also failed to take ownership of his sexist comment. He deleted the original ( which is why i had to link to a retweet ) obviously aware that what he had posted was wrong. Unfortunately for him, many people had already seen it and responded. You can\'t put the toothpaste back in the tube, tucker.\n[Document 2]\nTweet with a location you can add location information to your tweets, such as your city or precise location, from the web and via third-party applications. You always have the option to delete your tweet location history. Learn more\n[Document 3]\nI am not down with @karlrove @tuckercarlson misogynist mockery of @sarahpalinusa. Sick of it.\n[Document 4]\nOn monday night, while the rest of the world was watching charlie sheen flame out live on cnn, tucker carlson took to twitter to make some impolitic statements of his own. " palin\'s popularity falling in iowa, but maintains lead to become supreme commander of milfistan, " he wrote. By the next morning, the tweet was deleted and he had apologized, writing, "apparently charlie sheen got control of my twitter account last night while i was at dinner. Apologies for his behavior. " but that wasn\'t enough to spare him the ire of conservative women on the blogosphere and twitter. On tuesday, before carlson\'s first apology, stacy drake, writing on conservatives4palin, praised carlson\'s works at the daily caller, particularly the leaks of the journolist emails, saying that\'s why his tweet stung so badly. Aside from tucker\'s sheep-like response to warped poll numbers, he also failed to take ownership of his sexist comment. He deleted the original ( which is why i had to link to a retweet ) obviously aware that what he had posted was wrong. Unfortunately for him, many people had already seen it""",
        },
        {"role": "assistant", "content": "Document 2"
        },
        {"role": "user", "content": """[Summary]\nCalls to suicide hotlines have spiked dramatically since the deaths of kate spade and anthony bourdain — not an unusual phenomenon in the wake of celebrity suicides. The wall street journal reports on a 25 % uptick at the national suicide prevention lifeline ( 1-800-273-8255 ) since spade\'s death, while usa today attributed a similar rise to both celebs, and a new jersey hotline experienced a 70 % increase in calls friday morning. " we\'re so extremely busy, " says rachel larkin, who heads a crisis-prevention center in maryland. " i think we\'re all worried.... Both kate spade and anthony bourdain are people a lot of people related to. " celebrity suicides have been linked to suicide increases before. Marilyn monroe\'s 1962 death, for example, preceded a 12 % spike in suicides nationwide, the wrap reports. " when people who have suicidal thoughts see seemingly happy, famous and wealthy people dying of suicide, it makes them feel more hopeless, " says lauren foster, who heads a hotline in raleigh, north carolina. " they think if they died, what\'s to stop me? " but suicide-prevention advocates are emphasizing that hotlines do help and people\'s calls will go through, despite the surge this week. Check out buzzfeed to learn what happens on hotline calls and see what resources are available.\n\n[Document 1]\nSuicide hotlines can provide free and confidential support 24/7. Here\'s what to expect when you make the call. [ if you or someone you know is in immediate danger or having a medical emergency, call 911. ] the national suicide prevention lifeline ( 1-800-273-8255 ) is a toll-free hotline in the us for people in distress who feel like they are at risk of harming themselves. But what actually happens when you call? d3sign / getty images / via gettyimages.com suicide is complicated and sometimes hard to predict, but health experts say it can be preventable. That\'s why there are services like the national suicide\n[Document 2]\nHope. Love. Understanding. In a generation dealing with overwhelming issues like, depression, anxiety and relationship problems, the thehopeline is ready to meet you where you are, whatever your struggle. No judgement here. Just love. Through the dawson mcallister live radio show, one-on-one chat with a hopecoach, and resources on thehopeline.com, hope is available to anyone searching for it. Things can get better. Start your search for hope today and download our free ebook! understanding hope\n[Document 3]\nClose on average, there are 123 suicides per day in the united states. If you or someone you know needs help, call the national suicide prevention lifeline at 1-800-273-talk. Usa today anthony bourdain speaks during south by southwest at the austin convention center on sunday, march 13, 2016, in austin, texas. ( photo: rich fury, invision via ap ) the deaths of celebrity chef anthony bourdain and fashion designer kate spade this week have led to an uptick in calls to suicide prevention hotlines. The national suicide prevention lifeline saw a 25 % increase in volume over the last two days compared to the\n[Document 4]\nThe deaths of kate spade and anthony bourdain in a single week have led to a sharp increase in calls to suicide prevention hotlines. Publicity around the suicides of famous people has been linked to increases in suicide, and the phenomenon is nothing new: marilyn monroe\'s death in august 1962 was followed by a 12 percent increase in suicides nationwide, and 303 more people died than in august of the previous year, according to a study published in the journal of epidemiology &amp; community health. The thought of more people needing to call is upsetting, but at least indicates people are reaching out for help. The\n[Document 5]\nAs the world learned the news friday that renowned chef and food writer anthony bourdain had died by apparent suicide, the same phone number flooded the internet. The number for the national suicide prevention lifeline — 1-800-273-8255 — was pinned to the bottom of memorial instagram posts, shared in tweets and ran alongside news obituaries. Whenever...""",
        },
        {"role": "assistant", "content": "Document 2"
        },
        {"role": "user", "content": """[Summary]\nEducation secretary john king has a message for states where physical discipline is permitted in schools, per usa today: quit it. In a letter to governors and state school chiefs, king says 22 states — mostly in the south and west — still allow corporal punishment or don\'t forbid it. He implores them to stop the " harmful " and " ineffective " practice, saying it teaches kids that getting physical is ok to solve problems. He also points out that some corporal punishment taking place in schools would be considered criminal assault or battery in real-world settings. About 80 groups — including the naacp — lent their support to a similar letter penned monday by the national women\'s law center, reports cbs news. " corporal punishment of adults has been banned in prisons and in military training facilities, and it\'s time we do the same for our nation\'s schoolchildren, " an nwlc rep says. King also notes that physical punishment isn\'t applied equitably to all students. For example, even though black students make up about 16 % of attendees in public elementary and secondary schools, they\'re on the receiving end of one-third of the corporal punishment. Boys are subjected to 80 % of such acts, while students with disabilities also tend to be victims more so than other students. " these data and disparities shock the conscience, " king writes. ( alabama paddled 19k students in one school year. )\n\n[Document 1]\nWarczone is a collection of outsider-uploaded warcs, which are contributed to the internet archive but may or may not be ingested into the wayback machine. They are being kept in this location for reference and clarity for the wayback team, while also being accessible to the general public who are seeking any particular items they can regarding certain websites.\n[Document 2]\nNovember 21, 2016 an open letter to local and state educational agencies &amp; policymakers: on behalf of the national women\'s law center and the undersigned organizations and individuals, we call on local, state, and federal policymakers to address the damaging use of corporal punishment against our nation\'s schoolchildren. It is important to eliminate the use of corporal punishment in both public schools and private schools, which serve students receiving federal services, as well as assist in creating a safer learning environment for every child. Instead, we urge policymakers to ensure that our schools are places where students and educators interact in positive ways that foster students \' growth and dignity. More than 109,000 students were subjected to corporal punishment in public schools in the 2013-14 school year — down from\n[Document 3]\nBuffalo, n.y. -- education secretary john b. King jr. Is urging governors and school leaders in states that allow student paddling to end a practice he said would be considered "criminal assault or battery " against an adult. King released a letter tuesday asking leaders to replace corporal punishment with less punitive, more supportive disciplinary practices that he said work better against bad behavior. More than 110,000 students, including disproportionate numbers of black and disabled students, were subjected to paddling or a similar punishment in the 2013-14 school year, said king, citing the education department\'s civil rights data collection. Corporal punishment is legal in 22 states. "the practice has been clearly and repeatedly linked to negative health and academic outcomes for students, " king said during a conference call with reporters.\n[Document 4]\nEducation secretary john king speaks during the daily briefing at the white house in washington, thursday, sept. 29, 2016. ( ap photo/susan walsh ) ( photo: susan walsh, ap ) u.s. Education secretary john king is urging school districts nationwide to stop hitting and paddling students, saying corporal punishment is "harmful, ineffective, and often disproportionately applied to students of color and students with disabilities. " in a "dear colleague " letter being issued tuesday, king asks educators to "eliminate this practice from your schools, and instead promote supportive, effective disciplinary measures. "the use of corporal punishment can hinder the creation of a positive school climate by focusing on punitive measures to address student misbehavior rather than positive behavioral interventions and supports, " king writes. "corporal punishment also teaches students that""",
        },
        {"role": "assistant", "content": "Document 1"
        },
    ]

    return message_list

def get_cot_message() -> list:
    message_list = [
        {"role": "system", "content": "You are a helpful assistant.\n\
Your job is to identify the document that is not relevant to the summary.\n\
The relevance is determined by the mention of the key points in the summary.\n\
For example, the alert message from the twitter, the location information, and the unrelated information are considered irrelevant.\n\
Please response with the resaoning process and the result as document number or 'None'\n\
Do not end the response with a period.\n\
If multiple documents are irrelevant to the summary, please resonse in the form of 'Document 1|Document 3'\n"
        },
        {"role": "user", "content": "Given the summary and documents, please identify the document that is not relevant to the summary such as advertisements. If every document is relevant to the summary, output 'None'.\n\
Input document will be given in next dialogue.\n"
        },
        {"role": "user", "content": """[Summary]\nThe unemployment rate dropped to 8.2 % last month, but the economy only added 120,000 jobs, when 203,000 new jobs had been predicted, according to today\'s jobs report. Reaction on the wall street journal\'s marketbeat blog was swift: " woah!!! bad number. " the unemployment rate, however, is better news; it had been expected to hold steady at 8.3 %. But the ap notes that the dip is mostly due to more americans giving up on seeking employment.\n\n[Document 1]\nNational archives yes, it\'s that time again, folks. It\'s the first friday of the month, when for one ever-so-brief moment the interests of wall street, washington and main street are all aligned on one thing: jobs. A fresh update on the u.s. Employment situation for january hits the wires at 8: 30 a.m. New york time offering one of the most important snapshots on how the economy fared during the previous month. Expectations are for 203,000 new jobs to be created, according to economists polled by dow jones newswires, compared to 227,000 jobs added in february. The unemployment rate is expected to hold steady at 8.3 %. Here at marketbeat hq, we \' ll be offering color commentary before and after the data crosses the wires. Feel free to weigh-in yourself, via the comments section. And while you\'re here, why don\'t you sign up to. Enjoy the show.\n[Document 2]\nEmployers pulled back sharply on hiring last month, a reminder that the u.s. Economy may not be growing fast enough to sustain robust job growth. The unemployment rate dipped, but mostly because more americans stopped looking for work. The labor department says the economy added 120,000 jobs in march, down from more than 200,000 in each of the previous three months. The unemployment rate fell to 8.2 percent, the lowest since january 2009. The rate dropped because fewer people searched for jobs. The official unemployment tally only includes those seeking work. The economy has added 858,000 jobs since december _ the best four months of hiring in two years. But federal reserve chairman ben bernanke has cautioned that the current hiring pace is unlikely to continue without more consumer spending.""",
        },
        {"role": "assistant", "content": "The summary is about the unemployment rate and the economy report on the amount of newly added jobs. Document 1 mentions the unemployment rate. Document 2 mentions the unemployment rate and newly added jobs. Therefore, the irrelevant document is: All"
        },
        {"role": "user", "content": """[Summary]\nShelly sterling plans " eventually " to divorce her estranged husband donald, she tells barbara walters at abc news. As for her stake in the los angeles clippers, she plans to keep it, the ap notes. Sterling says she would " absolutely " fight any nba decision to force her to sell the team. The team is her " legacy " to her family, she says. " to be honest with you, i \' m wondering if a wife of one of the owners … said those racial slurs, would they oust the husband? or would they leave the husband in? "\n\n[Document 1]\nLos angeles ( ap ) — in her first interview since the nba banned her estranged husband, shelly sterling says she will fight to keep her share of the los angeles clippers and plans one day to divorce donald sterling. ( click prev or next to continue viewing images. ) advertisement ( click prev or next to continue viewing images. ) los angeles clippers co-owner shelly sterling, below, watches the clippers play the oklahoma city thunder along with her attorney, pierce o\'donnell, in the first half of game 3 of the western conference... ( associated press ) shelly sterling spoke to barbara walters, and abc news posted a short story with excerpts from the conversation sunday. Nba commissioner adam silver has banned donald sterling for making racist comments and urged owners to force sterling to sell the team. Silver added that no decisions had been made about the rest of sterling\'s family. According to abc\'s story, shelly sterling told walters: " i will fight that decision. " sterling also said that she " eventually " will divorce her husband, and that she hadn\'t yet done so due to financial considerations.\n[Document 2]\nShelly sterling said today that " eventually, i am going to " divorce her estranged husband, donald sterling, and if the nba tries to force her to sell her half of the los angeles clippers, she would " absolutely " fight to keep her stake in the team. " i will fight that decision, " she told abc news \' barbara walters today in an exclusive interview. " to be honest with you, i \' m wondering if a wife of one of the owners, and there\'s 30 owners, did something like that, said those racial slurs, would they oust the husband? or would they leave the husband in? " sterling added that the clippers franchise is her " passion " and " legacy to my family. " " i\'ve been with the team for 33 years, through the good times and the bad times, " she added. These comments come nearly two weeks after nba commissioner adam silver announced a lifetime ban and a $ 2.5 million fine for donald sterling on april 29, following racist comments from the 80-year-old, which were caught on tape and released to the media. Read: barbara walters \' exclusive interview with v. Stiviano being estranged from her husband, shelly sterling said she would " have to accept " whatever punishment the nba handed down to him, but that her stake in the team should be separate. " i was shocked by what he said. And -- well, i guess whatever their decision is -- we have to live""",
        },
        {"role": "assistant", "content": "The summary mentions Shelly Sterling's plans to eventually divorce her estranged husband Donald and her intention to keep her stake in the Los Angeles Clippers. Document 1 discusses Shelly Sterling's statements regarding divorce and her stake in the team. Document 2 also deals with the divorce of Shelly Sterling. Therefore, the irrelevant document is: All"
        },
        {"role": "user", "content": """[Summary]\nTucker carlson is in deep doodoo with conservative women after an ill-advised tweet referencing sarah palin that he posted, then removed, monday night. " palin\'s popularity falling in iowa, but maintains lead to become supreme commander of milfistan, " he tweeted — and we probably don\'t need to tell you where that is. His first attempt at an apology, which he tweeted the next morning: " apparently charlie sheen got control of my twitter account last night while i was at dinner. Apologies for his behavior. " that wasn\'t good enough for many conservative women, politico notes, rounding up reactions from bloggers to michelle malkin calling his behavior sexist and misogynistic. By late tuesday, carlson had offered up a more sincere-sounding apology: "i \' m sorry for last night\'s tweet. I meant absolutely no offense. Not the first dumb thing i\'ve said. Hopefully the last. " but at least one man — erick erickson, editor of redstate.com — was on carlson\'s side, tweeting his reaction to the post in question: " i laughed then got out my passport. "\n\n[Document 1]\nTucker carlson exposes his own sexism on twitter ( updated ) tucker carlson has done some good work in the past … his site, the daily caller, is a frequent stop of mine and many other conservatives. They were responsible for exposing the journolist scandal, which highlighted the planning and coordination of many members of the left-wing press. I will always be grateful to tucker\'s team for bringing that story to light. This is also why i am so angered by tucker\'s recent actions. I thought he was better than this. If you haven\'t heard by now, monday evening, tucker carlson posted a disturbing tweet about governor palin which said: palin\'s popularity falling in iowa, but maintains lead to become supreme commander of milfistan aside from tucker\'s sheep-like response to warped poll numbers, he also failed to take ownership of his sexist comment. He deleted the original ( which is why i had to link to a retweet ) obviously aware that what he had posted was wrong. Unfortunately for him, many people had already seen it and responded. You can\'t put the toothpaste back in the tube, tucker.\n[Document 2]\nTweet with a location you can add location information to your tweets, such as your city or precise location, from the web and via third-party applications. You always have the option to delete your tweet location history. Learn more\n[Document 3]\nI am not down with @karlrove @tuckercarlson misogynist mockery of @sarahpalinusa. Sick of it.\n[Document 4]\nOn monday night, while the rest of the world was watching charlie sheen flame out live on cnn, tucker carlson took to twitter to make some impolitic statements of his own. " palin\'s popularity falling in iowa, but maintains lead to become supreme commander of milfistan, " he wrote. By the next morning, the tweet was deleted and he had apologized, writing, "apparently charlie sheen got control of my twitter account last night while i was at dinner. Apologies for his behavior. " but that wasn\'t enough to spare him the ire of conservative women on the blogosphere and twitter. On tuesday, before carlson\'s first apology, stacy drake, writing on conservatives4palin, praised carlson\'s works at the daily caller, particularly the leaks of the journolist emails, saying that\'s why his tweet stung so badly. Aside from tucker\'s sheep-like response to warped poll numbers, he also failed to take ownership of his sexist comment. He deleted the original ( which is why i had to link to a retweet ) obviously aware that what he had posted was wrong. Unfortunately for him, many people had already seen it""",
        },
        {"role": "assistant", "content": "The summary discusses Tucker Carlson's controversial tweet regarding Sarah Palin and the subsequent backlash he faced from conservative women. Document 1 is related to Tucker Carlson's controversial tweet. Document 2 is unrelated as it contains information about adding location information to tweets and does not contribute to the discussion about Tucker Carlson's tweet and the reaction it received. Document 3 is a tweet message that calls Tucker Carlson a misogynist, related to the summary. Document 4 is related to the summary as it deals with Tucker Carlson's movement. Therefore, the irrelevant document is: Document 2"
        },
        # {"role": "user", "content": """[Summary]\nCalls to suicide hotlines have spiked dramatically since the deaths of kate spade and anthony bourdain — not an unusual phenomenon in the wake of celebrity suicides. The wall street journal reports on a 25 % uptick at the national suicide prevention lifeline ( 1-800-273-8255 ) since spade\'s death, while usa today attributed a similar rise to both celebs, and a new jersey hotline experienced a 70 % increase in calls friday morning. " we\'re so extremely busy, " says rachel larkin, who heads a crisis-prevention center in maryland. " i think we\'re all worried.... Both kate spade and anthony bourdain are people a lot of people related to. " celebrity suicides have been linked to suicide increases before. Marilyn monroe\'s 1962 death, for example, preceded a 12 % spike in suicides nationwide, the wrap reports. " when people who have suicidal thoughts see seemingly happy, famous and wealthy people dying of suicide, it makes them feel more hopeless, " says lauren foster, who heads a hotline in raleigh, north carolina. " they think if they died, what\'s to stop me? " but suicide-prevention advocates are emphasizing that hotlines do help and people\'s calls will go through, despite the surge this week. Check out buzzfeed to learn what happens on hotline calls and see what resources are available.\n\n[Document 1]\nSuicide hotlines can provide free and confidential support 24/7. Here\'s what to expect when you make the call. [ if you or someone you know is in immediate danger or having a medical emergency, call 911. ] the national suicide prevention lifeline ( 1-800-273-8255 ) is a toll-free hotline in the us for people in distress who feel like they are at risk of harming themselves. But what actually happens when you call? d3sign / getty images / via gettyimages.com suicide is complicated and sometimes hard to predict, but health experts say it can be preventable. That\'s why there are services like the national suicide\n[Document 2]\nHope. Love. Understanding. In a generation dealing with overwhelming issues like, depression, anxiety and relationship problems, the thehopeline is ready to meet you where you are, whatever your struggle. No judgement here. Just love. Through the dawson mcallister live radio show, one-on-one chat with a hopecoach, and resources on thehopeline.com, hope is available to anyone searching for it. Things can get better. Start your search for hope today and download our free ebook! understanding hope\n[Document 3]\nClose on average, there are 123 suicides per day in the united states. If you or someone you know needs help, call the national suicide prevention lifeline at 1-800-273-talk. Usa today anthony bourdain speaks during south by southwest at the austin convention center on sunday, march 13, 2016, in austin, texas. ( photo: rich fury, invision via ap ) the deaths of celebrity chef anthony bourdain and fashion designer kate spade this week have led to an uptick in calls to suicide prevention hotlines. The national suicide prevention lifeline saw a 25 % increase in volume over the last two days compared to the\n[Document 4]\nThe deaths of kate spade and anthony bourdain in a single week have led to a sharp increase in calls to suicide prevention hotlines. Publicity around the suicides of famous people has been linked to increases in suicide, and the phenomenon is nothing new: marilyn monroe\'s death in august 1962 was followed by a 12 percent increase in suicides nationwide, and 303 more people died than in august of the previous year, according to a study published in the journal of epidemiology &amp; community health. The thought of more people needing to call is upsetting, but at least indicates people are reaching out for help. The\n[Document 5]\nAs the world learned the news friday that renowned chef and food writer anthony bourdain had died by apparent suicide, the same phone number flooded the internet. The number for the national suicide prevention lifeline — 1-800-273-8255 — was pinned to the bottom of memorial instagram posts, shared in tweets and ran alongside news obituaries. Whenever...""",
        # },
        # {"role": "assistant", "content": "The summary discusses the increase in calls to suicide hotlines following the deaths of Kate Spade and Anthony Bourdain. Document 1 provides information about the National Suicide Prevention Lifeline and what to expect when calling a suicide hotline. Document 2 is not relevant to the summary as it discusses the services provided by TheHopeLine and does not contribute to the discussion about the spike in calls to suicide hotlines following the deaths of Kate Spade and Anthony Bourdain. Documents 3, 4, and 5 also contribute to the discussion about the increase in calls to suicide hotlines following celebrity suicides. Therefore, the irrelevant document is: Document 2"
        # },
        # {"role": "user", "content": """[Summary]\nEducation secretary john king has a message for states where physical discipline is permitted in schools, per usa today: quit it. In a letter to governors and state school chiefs, king says 22 states — mostly in the south and west — still allow corporal punishment or don\'t forbid it. He implores them to stop the " harmful " and " ineffective " practice, saying it teaches kids that getting physical is ok to solve problems. He also points out that some corporal punishment taking place in schools would be considered criminal assault or battery in real-world settings. About 80 groups — including the naacp — lent their support to a similar letter penned monday by the national women\'s law center, reports cbs news. " corporal punishment of adults has been banned in prisons and in military training facilities, and it\'s time we do the same for our nation\'s schoolchildren, " an nwlc rep says. King also notes that physical punishment isn\'t applied equitably to all students. For example, even though black students make up about 16 % of attendees in public elementary and secondary schools, they\'re on the receiving end of one-third of the corporal punishment. Boys are subjected to 80 % of such acts, while students with disabilities also tend to be victims more so than other students. " these data and disparities shock the conscience, " king writes. ( alabama paddled 19k students in one school year. )\n\n[Document 1]\nWarczone is a collection of outsider-uploaded warcs, which are contributed to the internet archive but may or may not be ingested into the wayback machine. They are being kept in this location for reference and clarity for the wayback team, while also being accessible to the general public who are seeking any particular items they can regarding certain websites.\n[Document 2]\nNovember 21, 2016 an open letter to local and state educational agencies &amp; policymakers: on behalf of the national women\'s law center and the undersigned organizations and individuals, we call on local, state, and federal policymakers to address the damaging use of corporal punishment against our nation\'s schoolchildren. It is important to eliminate the use of corporal punishment in both public schools and private schools, which serve students receiving federal services, as well as assist in creating a safer learning environment for every child. Instead, we urge policymakers to ensure that our schools are places where students and educators interact in positive ways that foster students \' growth and dignity. More than 109,000 students were subjected to corporal punishment in public schools in the 2013-14 school year — down from\n[Document 3]\nBuffalo, n.y. -- education secretary john b. King jr. Is urging governors and school leaders in states that allow student paddling to end a practice he said would be considered "criminal assault or battery " against an adult. King released a letter tuesday asking leaders to replace corporal punishment with less punitive, more supportive disciplinary practices that he said work better against bad behavior. More than 110,000 students, including disproportionate numbers of black and disabled students, were subjected to paddling or a similar punishment in the 2013-14 school year, said king, citing the education department\'s civil rights data collection. Corporal punishment is legal in 22 states. "the practice has been clearly and repeatedly linked to negative health and academic outcomes for students, " king said during a conference call with reporters.\n[Document 4]\nEducation secretary john king speaks during the daily briefing at the white house in washington, thursday, sept. 29, 2016. ( ap photo/susan walsh ) ( photo: susan walsh, ap ) u.s. Education secretary john king is urging school districts nationwide to stop hitting and paddling students, saying corporal punishment is "harmful, ineffective, and often disproportionately applied to students of color and students with disabilities. " in a "dear colleague " letter being issued tuesday, king asks educators to "eliminate this practice from your schools, and instead promote supportive, effective disciplinary measures. "the use of corporal punishment can hinder the creation of a positive school climate by focusing on punitive measures to address student misbehavior rather than positive behavioral interventions and supports, " king writes. "corporal punishment also teaches students that""",
        # },
        # {"role": "assistant", "content": "The summary discusses Education Secretary John King's message to states regarding the practice of physical discipline in schools. Document 1 talks about Warczone and outsider-uploaded warcs, which are not related to the topic of corporal punishment in schools addressed in the summary. Document 2, 3, and 4 is related to the summary as they mention John King and physical discipline. Therefore, the irrelevant document is: Document 1"
        # },
    ]

    return message_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file_name", type=str, default='raw_data', help="input file name")
    parser.add_argument("--output_file_name", type=str, default='cleansed_data', help="output file name")
    parser.add_argument("--start_idx", type=int, default=0, help="starting index of the data")
    parser.add_argument("--end_idx", type=int, default=10, help="ending index of the data")
    parser.add_argument("--model_version", type=str, default='gpt-3.5-turbo-0125')
    parser.add_argument("--num_agents", type=int, default=5, help="number of agents to use")
    parser.add_argument("--cleansing_mode", type=str, default='cot', help="zs, shot, cot")
    parser.add_argument("--error_patience", type=int, default=5, help="number of error allowed before stopping the program")
    parser.add_argument("--gpt_top_p", type=float, default=0.9, help="top_p value for GPT")
    parser.add_argument("--gpt_temperature", type=float, default=1, help="temperature value for GPT")
    parser.add_argument("--gpt_max_token", type=int, default=2048, help="max length value for GPT")
    parser.add_argument("--gpt_freq_penalty", type=float, default=0.05, help="frequency penalty value for GPT")
    parser.add_argument("--gpt_presence_penalty", type=float, default=0, help="presence penalty value for GPT")

    args = parser.parse_args()

    main(args)
