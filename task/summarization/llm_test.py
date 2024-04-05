# Standard Library Modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import logging
import argparse
# 3rd-party Modules
import pandas as pd
from tqdm.auto import tqdm
from nlgeval import NLGEval
from bert_score import BERTScorer
from BARTScore.bart_score import BARTScorer
from rouge_score import rouge_scorer
# Pytorch Modules
import torch
torch.set_num_threads(2)
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Huggingface Modules
from transformers import AutoModelForCausalLM, AutoTokenizer
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.summarization.dataset import SummarizationDataset, collate_fn
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_wandb_exp_name, get_torch_device, check_path, get_huggingface_model_name

def testing(args: argparse.Namespace) -> None:
    device = get_torch_device(args.device)

    # Define logger
    logger = logging.getLogger(__name__)
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    # Initialize tensorboard writer
    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.log_path, get_tb_exp_name(args)))
        writer.add_text('args', str(args))

    # Load dataset and define dataloader
    write_log(logger, "Loading dataset...")
    dataset_test = SummarizationDataset(args, os.path.join(args.preprocess_path, args.task, args.test_dataset, 'test.pkl'), 'test') # For fair comparison, share multi_news_plus test data for all models
    dataset_test.data_list = dataset_test.data_list[:100] # Set the number of samples to 100 for quick testing
    dataloader_test = DataLoader(dataset_test, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, drop_last=False, collate_fn=collate_fn)

    # Get model instance
    write_log(logger, "Building model")
    model_name = get_huggingface_model_name(args.model_type)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load Wandb
    if args.use_wandb:
        import wandb
        from wandb import AlertLevel
        wandb.init(project=args.proj_name,
                   name=get_wandb_exp_name(args),
                   config=args,
                   notes=args.description,
                   tags=["TEST",
                         f"Dataset: {args.task_dataset}",
                         f"Model: {args.model_type}"])

    # Test - Start evaluation
    model = model.eval()
    result_list = []
    ref_list = []
    hyp_list = []

    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_test, total=len(dataloader_test), desc=f'Testing')):
        # Test - Get data from batch
        source_text = data_dicts['source_text']
        target_text = data_dicts['target_text']

        input_prompt = build_prompt(args, source_text)
        model_inputs = tokenizer.apply_chat_template(input_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_tokens = model.generate(model_inputs, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.9,)
        generated_target = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        generated_target = postprocess_output(args, generated_target)

        for idx, (source, target, generated) in enumerate(zip(source_text, target_text, generated_target)):
            result_list.append({
                'source': source,
                'target': target,
                'generated': generated,
            })

        ref_list.extend([target_text])
        hyp_list.extend([generated_target])
    ref_list = [list(map(str, refs)) for refs in zip(*ref_list)]

    # Delete model to save memory
    model = model.to('cpu')
    del model

    # Test - nlg-eval
    write_log(logger, "TEST - Calculating NLG-eval metrics...")
    Eval = NLGEval(metrics_to_omit=['SPICE', 'CIDEr', 'SkipThoughtCS', 'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity', 'GreedyMatchingScore'])
    BERT_Eval = BERTScorer(device=args.device, model_type='bert-base-uncased')
    BART_Eval = BARTScorer(device=args.device, checkpoint='facebook/bart-large-cnn')
    ROUGE_Eval = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)

    # I don't know why but we need this
    _strip = lambda x: x.strip()
    ref_list2 = [list(map(_strip, refs)) for refs in zip(*ref_list)]


    # print(len(ref_list2), len(hyp_list), ref_list2, hyp_list)

    #print(ref_list)

    metrics_dict = Eval.compute_metrics(ref_list, hyp_list)

    # Test - ROUGE
    rouge_1, rouge_2, rouge_L, rouge_Lsum = 0, 0, 0, 0
    write_log(logger, "TEST - Calculating ROUGE metrics...")
    for idx, (ref, hyp) in enumerate(zip(ref_list, hyp_list)):
        ref = ref[0]
        hyp = hyp[0]

        scores = ROUGE_Eval.score(ref, hyp)
        rouge_1 += scores['rouge1'].fmeasure
        rouge_2 += scores['rouge2'].fmeasure
        rouge_L += scores['rougeL'].fmeasure
        rouge_Lsum += scores['rougeLsum'].fmeasure
    rouge_1 /= len(ref_list)
    rouge_2 /= len(ref_list)
    rouge_L /= len(ref_list)
    rouge_Lsum /= len(ref_list)

    # Test - BERTScore and BARTScore
    bert_score_P, bert_score_R, bert_score_F1, bart_score_total = 0, 0, 0, 0
    # List of list -> List of str

    ref_list = ref_list[0]
    print(ref_list)
    print(hyp_list)

    write_log(logger, "TEST - Calculating BERTScore metrics...")
    bert_score_P, bert_score_R, bert_score_F1 = BERT_Eval.score(ref_list, hyp_list, batch_size=args.test_batch_size)

    write_log(logger, "TEST - Calculating BARTScore metrics...")
    bart_score_total = BART_Eval.score(ref_list, hyp_list, batch_size=args.test_batch_size)

    bert_score_P = bert_score_P.mean().item()
    bert_score_R = bert_score_R.mean().item()
    bert_score_F1 = bert_score_F1.mean().item()
    bart_score_total = sum(bart_score_total) / len(bart_score_total)

    # Final - End of testing
    write_log(logger, f"TEST - Bleu_1: {metrics_dict['Bleu_1']:.4f}")
    write_log(logger, f"TEST - Bleu_2: {metrics_dict['Bleu_2']:.4f}")
    write_log(logger, f"TEST - Bleu_3: {metrics_dict['Bleu_3']:.4f}")
    write_log(logger, f"TEST - Bleu_4: {metrics_dict['Bleu_4']:.4f}")
    write_log(logger, f"TEST - Bleu_avg: {(metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4:.4f}")
    write_log(logger, f"TEST - Rouge_1: {rouge_1:.4f}")
    write_log(logger, f"TEST - Rouge_2: {rouge_2:.4f}")
    write_log(logger, f"TEST - Rouge_L: {rouge_L:.4f}")
    write_log(logger, f"TEST - Rouge_Lsum: {rouge_Lsum:.4f}")
    write_log(logger, f"TEST - Rouge_L_NLGEVAL: {metrics_dict['ROUGE_L']:.4f}")
    write_log(logger, f"TEST - Meteor: {metrics_dict['METEOR']:.4f}")
    write_log(logger, f"TEST - BERTScore_Precision: {bert_score_P:.4f}")
    write_log(logger, f"TEST - BERTScore_Recall: {bert_score_R:.4f}")
    write_log(logger, f"TEST - BERTScore_F1: {bert_score_F1:.4f}")
    write_log(logger, f"TEST - BARTScore: {bart_score_total:.4f}")

    # Save data as json file
    save_path = os.path.join(args.result_path, args.task, args.task_dataset)
    check_path(save_path)

    result_dict = {
        'args': vars(args),
        'Bleu_1': metrics_dict['Bleu_1'],
        'Bleu_2': metrics_dict['Bleu_2'],
        'Bleu_3': metrics_dict['Bleu_3'],
        'Bleu_4': metrics_dict['Bleu_4'],
        'Bleu_avg': (metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4,
        'Rouge_1': rouge_1,
        'Rouge_2': rouge_2,
        'Rouge_L': rouge_L,
        'Rouge_Lsum': rouge_Lsum,
        'Rouge_L_NLG': metrics_dict['ROUGE_L'],
        'Meteor': metrics_dict['METEOR'],
        'BERTScore_Precision': bert_score_P,
        'BERTScore_Recall': bert_score_R,
        'BERTScore_F1': bert_score_F1,
        'BARTScore': bart_score_total,
        'result_list': result_list,
    }
    save_name = os.path.join(save_path, f'test_result_{args.model_type}.json')
    with open(save_name, 'w') as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    if args.use_tensorboard:
        writer.add_scalar('TEST/Bleu_1', metrics_dict['Bleu_1'], global_step=0)
        writer.add_scalar('TEST/Bleu_2', metrics_dict['Bleu_2'], global_step=0)
        writer.add_scalar('TEST/Bleu_3', metrics_dict['Bleu_3'], global_step=0)
        writer.add_scalar('TEST/Bleu_4', metrics_dict['Bleu_4'], global_step=0)
        writer.add_scalar('TEST/Bleu_avg', (metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4, global_step=0)
        writer.add_scalar('TEST/Rouge_1', rouge_1, global_step=0)
        writer.add_scalar('TEST/Rouge_2', rouge_2, global_step=0)
        writer.add_scalar('TEST/Rouge_L', rouge_L, global_step=0)
        writer.add_scalar('TEST/Rouge_Lsum', rouge_Lsum, global_step=0)
        writer.add_scalar('TEST/Rouge_L_NLG', metrics_dict['ROUGE_L'], global_step=0)
        writer.add_scalar('TEST/Meteor', metrics_dict['METEOR'], global_step=0)
        writer.add_scalar('TEST/BERTScore_Precision', bert_score_P, global_step=0)
        writer.add_scalar('TEST/BERTScore_Recall', bert_score_R, global_step=0)
        writer.add_scalar('TEST/BERTScore_F1', bert_score_F1, global_step=0)
        writer.add_scalar('TEST/BARTScore', bart_score_total, global_step=0)

        writer.close()
    if args.use_wandb:
        wandb_df = pd.DataFrame({
            'Dataset': [args.task_dataset],
            'Test_Dataset': [args.test_dataset],
            'Model': [args.model_type],
            'Bleu_1': [metrics_dict['Bleu_1']],
            'Bleu_2': [metrics_dict['Bleu_2']],
            'Bleu_3': [metrics_dict['Bleu_3']],
            'Bleu_4': [metrics_dict['Bleu_4']],
            'Bleu_avg': [(metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4],
            'Rouge_1': [rouge_1],
            'Rouge_2': [rouge_2],
            'Rouge_L': [rouge_L],
            'Rouge_Lsum': [rouge_Lsum],
            'Rouge_L_NLG': [metrics_dict['ROUGE_L']],
            'Meteor': [metrics_dict['METEOR']],
            'BERTScore_Precision': [bert_score_P],
            'BERTScore_Recall': [bert_score_R],
            'BERTScore_F1': [bert_score_F1],
            'BARTScore': [bart_score_total],
        })
        wandb_table = wandb.Table(dataframe=wandb_df)
        wandb.log({"TEST_Result": wandb_table})
        wandb.save(save_name)

        wandb.finish()

    return metrics_dict

def build_prompt(args: argparse.Namespace, source_text: str) -> str:
    # Split concatenated multi documents using args.doc_sep_token
    source_text = source_text[0]
    source_list = source_text.split(args.doc_sep_token)

    if args.task_dataset == 'multi_news_llm_noise0':
        default_messages = [
            {"role": "user", "content": "Summarize given documents into one summary:\n\n\
[Document 1]\n\
Tucker carlson exposes his own sexism on twitter ( updated ) tucker carlson has done some good work in the past … his site, the daily caller, is a frequent stop of mine and many other conservatives. They were responsible for exposing the journolist scandal, which highlighted the planning and coordination of many members of the left-wing press. I will always be grateful to tucker's team for bringing that story to light. This is also why i am so angered by tucker's recent actions. I thought he was better than this. If you haven't heard by now, monday evening, tucker carlson posted a disturbing tweet about governor palin which said: palin's popularity falling in iowa, but maintains lead to become supreme commander of milfistan aside from tucker's sheep-like response to warped poll numbers, he also failed to take ownership of his sexist comment. He deleted the original ( which is why i had to link to a retweet ) obviously aware that what he had posted was wrong. Unfortunately for him, many people had already seen it and responded. You can't put the toothpaste back in the tube, tucker.\n\
[Document 2]\n\
I am not down with @karlrove @tuckercarlson misogynist mockery of @sarahpalinusa. Sick of it.\n\
[Document 3]\n\
On monday night, while the rest of the world was watching charlie sheen flame out live on cnn, tucker carlson took to twitter to make some impolitic statements of his own. \" palin's popularity falling in iowa, but maintains lead to become supreme commander of milfistan, \" he wrote. By the next morning, the tweet was deleted and he had apologized, writing, \"apparently charlie sheen got control of my twitter account last night while i was at dinner. Apologies for his behavior. \" but that wasn't enough to spare him the ire of conservative women on the blogosphere and twitter. On tuesday, before carlson's first apology, stacy drake, writing on conservatives4palin, praised carlson's works at the daily caller, particularly the leaks of the journolist emails, saying that's why his tweet stung so badly. Aside from tucker's sheep-like response to warped poll numbers, he also failed to take ownership of his sexist comment. He deleted the original ( which is why i had to link to a retweet ) obviously aware that what he had posted was wrong. Unfortunately for him, many people had already seen it"},
            {"role": "assistant", "content": "[Summary]\n\
Tucker carlson is in deep doodoo with conservative women after an ill-advised tweet referencing sarah palin that he posted, then removed, monday night. \" palin's popularity falling in iowa, but maintains lead to become supreme commander of milfistan, \" he tweeted — and we probably don't need to tell you where that is. His first attempt at an apology, which he tweeted the next morning: \" apparently charlie sheen got control of my twitter account last night while i was at dinner. Apologies for his behavior. \" that wasn't good enough for many conservative women, politico notes, rounding up reactions from bloggers to michelle malkin calling his behavior sexist and misogynistic. By late tuesday, carlson had offered up a more sincere-sounding apology: \"i ' m sorry for last night's tweet. I meant absolutely no offense. Not the first dumb thing i've said. Hopefully the last. \" but at least one man — erick erickson, editor of redstate.com — was on carlson's side, tweeting his reaction to the post in question: \" i laughed then got out my passport. \""},
            {"role": "user", "content": "Summarize given documents into one summary:\n\n\
[Document 1]\n\
November 21, 2016 an open letter to local and state educational agencies &amp; policymakers: on behalf of the national women's law center and the undersigned organizations and individuals, we call on local, state, and federal policymakers to address the damaging use of corporal punishment against our nation's schoolchildren. It is important to eliminate the use of corporal punishment in both public schools and private schools, which serve students receiving federal services, as well as assist in creating a safer learning environment for every child. Instead, we urge policymakers to ensure that our schools are places where students and educators interact in positive ways that foster students ' growth and dignity. More than 109,000 students were subjected to corporal punishment in public schools in the 2013-14 school year — down from\n\
[Document 2]\n\
Buffalo, n.y. -- education secretary john b. King jr. Is urging governors and school leaders in states that allow student paddling to end a practice he said would be considered \"criminal assault or battery \" against an adult. King released a letter tuesday asking leaders to replace corporal punishment with less punitive, more supportive disciplinary practices that he said work better against bad behavior. More than 110,000 students, including disproportionate numbers of black and disabled students, were subjected to paddling or a similar punishment in the 2013-14 school year, said king, citing the education department's civil rights data collection. Corporal punishment is legal in 22 states. \"the practice has been clearly and repeatedly linked to negative health and academic outcomes for students, \" king said during a conference call with reporters.\n\
[Document 3]\n\
Education secretary john king speaks during the daily briefing at the white house in washington, thursday, sept. 29, 2016. ( ap photo/susan walsh ) ( photo: susan walsh, ap ) u.s. Education secretary john king is urging school districts nationwide to stop hitting and paddling students, saying corporal punishment is \"harmful, ineffective, and often disproportionately applied to students of color and students with disabilities. \" in a \"dear colleague \" letter being issued tuesday, king asks educators to \"eliminate this practice from your schools, and instead promote supportive, effective disciplinary measures. \"the use of corporal punishment can hinder the creation of a positive school climate by focusing on punitive measures to address student misbehavior rather than positive behavioral interventions and supports, \" king writes. \"corporal punishment also teaches students that"},
            {"role": "assistant", "content": "[Summary]\n\
Education secretary john king has a message for states where physical discipline is permitted in schools, per usa today: quit it. In a letter to governors and state school chiefs, king says 22 states — mostly in the south and west — still allow corporal punishment or don't forbid it. He implores them to stop the \" harmful \" and \" ineffective \" practice, saying it teaches kids that getting physical is ok to solve problems. He also points out that some corporal punishment taking place in schools would be considered criminal assault or battery in real-world settings. About 80 groups — including the naacp — lent their support to a similar letter penned monday by the national women's law center, reports cbs news. \" corporal punishment of adults has been banned in prisons and in military training facilities, and it's time we do the same for our nation's schoolchildren, \" an nwlc rep says. King also notes that physical punishment isn't applied equitably to all students. For example, even though black students make up about 16 % of attendees in public elementary and secondary schools, they're on the receiving end of one-third of the corporal punishment. Boys are subjected to 80 % of such acts, while students with disabilities also tend to be victims more so than other students. \" these data and disparities shock the conscience, \" king writes. ( alabama paddled 19k students in one school year. )"},
        ]
    elif args.task_dataset == 'multi_news_llm_noise1':
        default_messages = [
            {"role": "user", "content": "Summarize given documents into one summary:\n\n\
[Document 1]\n\
Tucker carlson exposes his own sexism on twitter ( updated ) tucker carlson has done some good work in the past … his site, the daily caller, is a frequent stop of mine and many other conservatives. They were responsible for exposing the journolist scandal, which highlighted the planning and coordination of many members of the left-wing press. I will always be grateful to tucker's team for bringing that story to light. This is also why i am so angered by tucker's recent actions. I thought he was better than this. If you haven't heard by now, monday evening, tucker carlson posted a disturbing tweet about governor palin which said: palin's popularity falling in iowa, but maintains lead to become supreme commander of milfistan aside from tucker's sheep-like response to warped poll numbers, he also failed to take ownership of his sexist comment. He deleted the original ( which is why i had to link to a retweet ) obviously aware that what he had posted was wrong. Unfortunately for him, many people had already seen it and responded. You can't put the toothpaste back in the tube, tucker.\n\
[Document 2]\n\
I am not down with @karlrove @tuckercarlson misogynist mockery of @sarahpalinusa. Sick of it.\n\
[Document 3]\n\
On monday night, while the rest of the world was watching charlie sheen flame out live on cnn, tucker carlson took to twitter to make some impolitic statements of his own. \" palin's popularity falling in iowa, but maintains lead to become supreme commander of milfistan, \" he wrote. By the next morning, the tweet was deleted and he had apologized, writing, \"apparently charlie sheen got control of my twitter account last night while i was at dinner. Apologies for his behavior. \" but that wasn't enough to spare him the ire of conservative women on the blogosphere and twitter. On tuesday, before carlson's first apology, stacy drake, writing on conservatives4palin, praised carlson's works at the daily caller, particularly the leaks of the journolist emails, saying that's why his tweet stung so badly. Aside from tucker's sheep-like response to warped poll numbers, he also failed to take ownership of his sexist comment. He deleted the original ( which is why i had to link to a retweet ) obviously aware that what he had posted was wrong. Unfortunately for him, many people had already seen it"},
            {"role": "assistant", "content": "[Summary]\n\
Tucker carlson is in deep doodoo with conservative women after an ill-advised tweet referencing sarah palin that he posted, then removed, monday night. \" palin's popularity falling in iowa, but maintains lead to become supreme commander of milfistan, \" he tweeted — and we probably don't need to tell you where that is. His first attempt at an apology, which he tweeted the next morning: \" apparently charlie sheen got control of my twitter account last night while i was at dinner. Apologies for his behavior. \" that wasn't good enough for many conservative women, politico notes, rounding up reactions from bloggers to michelle malkin calling his behavior sexist and misogynistic. By late tuesday, carlson had offered up a more sincere-sounding apology: \"i ' m sorry for last night's tweet. I meant absolutely no offense. Not the first dumb thing i've said. Hopefully the last. \" but at least one man — erick erickson, editor of redstate.com — was on carlson's side, tweeting his reaction to the post in question: \" i laughed then got out my passport. \""},
            {"role": "user", "content": "Summarize given documents into one summary:\n\n\
[Document 1]\n\
Warczone is a collection of outsider-uploaded warcs, which are contributed to the internet archive but may or may not be ingested into the wayback machine. They are being kept in this location for reference and clarity for the wayback team, while also being accessible to the general public who are seeking any particular items they can regarding certain websites.\n\
[Document 2]\n\
November 21, 2016 an open letter to local and state educational agencies &amp; policymakers: on behalf of the national women's law center and the undersigned organizations and individuals, we call on local, state, and federal policymakers to address the damaging use of corporal punishment against our nation's schoolchildren. It is important to eliminate the use of corporal punishment in both public schools and private schools, which serve students receiving federal services, as well as assist in creating a safer learning environment for every child. Instead, we urge policymakers to ensure that our schools are places where students and educators interact in positive ways that foster students ' growth and dignity. More than 109,000 students were subjected to corporal punishment in public schools in the 2013-14 school year — down from\n\
[Document 3]\n\
Buffalo, n.y. -- education secretary john b. King jr. Is urging governors and school leaders in states that allow student paddling to end a practice he said would be considered \"criminal assault or battery \" against an adult. King released a letter tuesday asking leaders to replace corporal punishment with less punitive, more supportive disciplinary practices that he said work better against bad behavior. More than 110,000 students, including disproportionate numbers of black and disabled students, were subjected to paddling or a similar punishment in the 2013-14 school year, said king, citing the education department's civil rights data collection. Corporal punishment is legal in 22 states. \"the practice has been clearly and repeatedly linked to negative health and academic outcomes for students, \" king said during a conference call with reporters.\n\
[Document 4]\n\
Education secretary john king speaks during the daily briefing at the white house in washington, thursday, sept. 29, 2016. ( ap photo/susan walsh ) ( photo: susan walsh, ap ) u.s. Education secretary john king is urging school districts nationwide to stop hitting and paddling students, saying corporal punishment is \"harmful, ineffective, and often disproportionately applied to students of color and students with disabilities. \" in a \"dear colleague \" letter being issued tuesday, king asks educators to \"eliminate this practice from your schools, and instead promote supportive, effective disciplinary measures. \"the use of corporal punishment can hinder the creation of a positive school climate by focusing on punitive measures to address student misbehavior rather than positive behavioral interventions and supports, \" king writes. \"corporal punishment also teaches students that"},
            {"role": "assistant", "content": "[Summary]\n\
Education secretary john king has a message for states where physical discipline is permitted in schools, per usa today: quit it. In a letter to governors and state school chiefs, king says 22 states — mostly in the south and west — still allow corporal punishment or don't forbid it. He implores them to stop the \" harmful \" and \" ineffective \" practice, saying it teaches kids that getting physical is ok to solve problems. He also points out that some corporal punishment taking place in schools would be considered criminal assault or battery in real-world settings. About 80 groups — including the naacp — lent their support to a similar letter penned monday by the national women's law center, reports cbs news. \" corporal punishment of adults has been banned in prisons and in military training facilities, and it's time we do the same for our nation's schoolchildren, \" an nwlc rep says. King also notes that physical punishment isn't applied equitably to all students. For example, even though black students make up about 16 % of attendees in public elementary and secondary schools, they're on the receiving end of one-third of the corporal punishment. Boys are subjected to 80 % of such acts, while students with disabilities also tend to be victims more so than other students. \" these data and disparities shock the conscience, \" king writes. ( alabama paddled 19k students in one school year. )"},
        ]
    elif args.task_dataset == 'multi_news_llm_noise2':
        default_messages = [
            {"role": "user", "content": "Summarize given documents into one summary:\n\n\
[Document 1]\n\
Tucker carlson exposes his own sexism on twitter ( updated ) tucker carlson has done some good work in the past … his site, the daily caller, is a frequent stop of mine and many other conservatives. They were responsible for exposing the journolist scandal, which highlighted the planning and coordination of many members of the left-wing press. I will always be grateful to tucker's team for bringing that story to light. This is also why i am so angered by tucker's recent actions. I thought he was better than this. If you haven't heard by now, monday evening, tucker carlson posted a disturbing tweet about governor palin which said: palin's popularity falling in iowa, but maintains lead to become supreme commander of milfistan aside from tucker's sheep-like response to warped poll numbers, he also failed to take ownership of his sexist comment. He deleted the original ( which is why i had to link to a retweet ) obviously aware that what he had posted was wrong. Unfortunately for him, many people had already seen it and responded. You can't put the toothpaste back in the tube, tucker.\n\
[Document 2]\n\
Tweet with a location you can add location information to your tweets, such as your city or precise location, from the web and via third-party applications. You always have the option to delete your tweet location history. Learn more\n\
[Document 3]\n\
I am not down with @karlrove @tuckercarlson misogynist mockery of @sarahpalinusa. Sick of it.\n\
[Document 4]\n\
On monday night, while the rest of the world was watching charlie sheen flame out live on cnn, tucker carlson took to twitter to make some impolitic statements of his own. \" palin's popularity falling in iowa, but maintains lead to become supreme commander of milfistan, \" he wrote. By the next morning, the tweet was deleted and he had apologized, writing, \"apparently charlie sheen got control of my twitter account last night while i was at dinner. Apologies for his behavior. \" but that wasn't enough to spare him the ire of conservative women on the blogosphere and twitter. On tuesday, before carlson's first apology, stacy drake, writing on conservatives4palin, praised carlson's works at the daily caller, particularly the leaks of the journolist emails, saying that's why his tweet stung so badly. Aside from tucker's sheep-like response to warped poll numbers, he also failed to take ownership of his sexist comment. He deleted the original ( which is why i had to link to a retweet ) obviously aware that what he had posted was wrong. Unfortunately for him, many people had already seen it"},
            {"role": "assistant", "content": "[Summary]\n\
Tucker carlson is in deep doodoo with conservative women after an ill-advised tweet referencing sarah palin that he posted, then removed, monday night. \" palin's popularity falling in iowa, but maintains lead to become supreme commander of milfistan, \" he tweeted — and we probably don't need to tell you where that is. His first attempt at an apology, which he tweeted the next morning: \" apparently charlie sheen got control of my twitter account last night while i was at dinner. Apologies for his behavior. \" that wasn't good enough for many conservative women, politico notes, rounding up reactions from bloggers to michelle malkin calling his behavior sexist and misogynistic. By late tuesday, carlson had offered up a more sincere-sounding apology: \"i ' m sorry for last night's tweet. I meant absolutely no offense. Not the first dumb thing i've said. Hopefully the last. \" but at least one man — erick erickson, editor of redstate.com — was on carlson's side, tweeting his reaction to the post in question: \" i laughed then got out my passport. \""},
            {"role": "user", "content": "Summarize given documents into one summary:\n\n\
[Document 1]\n\
Warczone is a collection of outsider-uploaded warcs, which are contributed to the internet archive but may or may not be ingested into the wayback machine. They are being kept in this location for reference and clarity for the wayback team, while also being accessible to the general public who are seeking any particular items they can regarding certain websites.\n\
[Document 2]\n\
November 21, 2016 an open letter to local and state educational agencies &amp; policymakers: on behalf of the national women's law center and the undersigned organizations and individuals, we call on local, state, and federal policymakers to address the damaging use of corporal punishment against our nation's schoolchildren. It is important to eliminate the use of corporal punishment in both public schools and private schools, which serve students receiving federal services, as well as assist in creating a safer learning environment for every child. Instead, we urge policymakers to ensure that our schools are places where students and educators interact in positive ways that foster students ' growth and dignity. More than 109,000 students were subjected to corporal punishment in public schools in the 2013-14 school year — down from\n\
[Document 3]\n\
Buffalo, n.y. -- education secretary john b. King jr. Is urging governors and school leaders in states that allow student paddling to end a practice he said would be considered \"criminal assault or battery \" against an adult. King released a letter tuesday asking leaders to replace corporal punishment with less punitive, more supportive disciplinary practices that he said work better against bad behavior. More than 110,000 students, including disproportionate numbers of black and disabled students, were subjected to paddling or a similar punishment in the 2013-14 school year, said king, citing the education department's civil rights data collection. Corporal punishment is legal in 22 states. \"the practice has been clearly and repeatedly linked to negative health and academic outcomes for students, \" king said during a conference call with reporters.\n\
[Document 4]\n\
Education secretary john king speaks during the daily briefing at the white house in washington, thursday, sept. 29, 2016. ( ap photo/susan walsh ) ( photo: susan walsh, ap ) u.s. Education secretary john king is urging school districts nationwide to stop hitting and paddling students, saying corporal punishment is \"harmful, ineffective, and often disproportionately applied to students of color and students with disabilities. \" in a \"dear colleague \" letter being issued tuesday, king asks educators to \"eliminate this practice from your schools, and instead promote supportive, effective disciplinary measures. \"the use of corporal punishment can hinder the creation of a positive school climate by focusing on punitive measures to address student misbehavior rather than positive behavioral interventions and supports, \" king writes. \"corporal punishment also teaches students that"},
            {"role": "assistant", "content": "[Summary]\n\
Education secretary john king has a message for states where physical discipline is permitted in schools, per usa today: quit it. In a letter to governors and state school chiefs, king says 22 states — mostly in the south and west — still allow corporal punishment or don't forbid it. He implores them to stop the \" harmful \" and \" ineffective \" practice, saying it teaches kids that getting physical is ok to solve problems. He also points out that some corporal punishment taking place in schools would be considered criminal assault or battery in real-world settings. About 80 groups — including the naacp — lent their support to a similar letter penned monday by the national women's law center, reports cbs news. \" corporal punishment of adults has been banned in prisons and in military training facilities, and it's time we do the same for our nation's schoolchildren, \" an nwlc rep says. King also notes that physical punishment isn't applied equitably to all students. For example, even though black students make up about 16 % of attendees in public elementary and secondary schools, they're on the receiving end of one-third of the corporal punishment. Boys are subjected to 80 % of such acts, while students with disabilities also tend to be victims more so than other students. \" these data and disparities shock the conscience, \" king writes. ( alabama paddled 19k students in one school year. )"},
        ]

    # append source text to default messages
    default_messages.append({"role": "user", "content": "Summarize given documents into one summary:\n\n"})
    for idx, source in enumerate(source_list):
        default_messages[-1]["content"] += f"[Document {idx+1}]\n{source}\n"

    return default_messages

def postprocess_output(args: argparse.Namespace, output: str) -> str:
    # Postprocess the output

    if args.model_type == 'gemma':
        model_response = output[0].split('<end_of_turn>')[-1]
        model_response = model_response.split('\n')[-1]
        model_response = model_response.replace('<eos>', '')
    elif args.model_type in ['mistral', 'llama2']:
        model_response = output[0].split('[/INST]')[-1]
        model_response = model_response.strip()
        model_response = model_response.replace('</s>', '')

    return model_response
