from util import process_story
from glob import glob 
from tqdm import tqdm

OUTFILE_SUMMARY = "data/cnndm/{}.cnndm_source"
OUTFILE_ARTICLE = "data/cnndm/{}.cnndm_target"
SPLIT_RATIO = {"train" : 0.8, "valid":0.1, "test": 0.1} # train/valid/test

if __name__ == "__main__":
    paths = glob("data/cnndm/cnn/stories/*")

    summary_lst = []
    article_lst = []
    error_paths = []
    print("Parsing cnn/dm...")
    for path in tqdm(paths):
        with open(path,'rt',encoding='utf8') as f:
            sample = f.read()
        story, summary = process_story(sample)
        
        # remove article origin i.e. (CNN) -- 
        story_rm_prefix = " ".join(story).split(" -- ")
        story_rm_prefix = story_rm_prefix[1:] if len(story_rm_prefix) != 1 else story_rm_prefix
        story_rm_prefix = " -- ".join(story_rm_prefix)

        # remove remaining article origin i.e. new delhi(CNN) -> in case (CNN) is in the middle, text is much lost
        # story_rm_prefix = story_rm_prefix.split("(CNN)")
        # story_rm_prefix = story_rm_prefix[1:] if len(story_rm_prefix) != 1 else story_rm_prefix
        # story_rm_prefix = "(CNN)".join(story_rm_prefix)

        # story_rm_prefix = " ".join(story)
        summary = " ".join(summary)

        if not story_rm_prefix.strip(): 
            error_paths.append(path)
            continue

        summary_lst.append(summary.strip())
        article_lst.append(story_rm_prefix.strip())

    with open("error.out",'w') as ferr:
        ferr.write("NO STORYS FOR SAMPLES BELOW!\n")
        ferr.write("\n".join(error_paths))
    print(f"No Story for {len(error_paths)} sapmles : see error.out for details!")

    print("Writing Results...")
    start_idx = 0
    for split_name, ratio in SPLIT_RATIO.items():
        end_idx = min(start_idx + round(len(summary_lst) * ratio), len(summary_lst))
        with open(OUTFILE_SUMMARY.format(split_name), 'w') as fsum, open(OUTFILE_ARTICLE.format(split_name),'w') as fart:
            fsum.write("\n".join(summary_lst[start_idx : end_idx])+'\n')
            fart.write("\n".join(article_lst[start_idx : end_idx])+'\n')
        start_idx = end_idx    

    
