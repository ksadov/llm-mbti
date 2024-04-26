import json
import os
import math
import seaborn as sns
import matplotlib.pyplot as plt

VALID_MBTI_TYPES = [
    "INTP", "ENTP", "ISTP", "ESTP", "ISTJ", "ESTJ", "INFP", "ENFP", "ESFP", "ISFJ", "ISFP", "ESFJ", "ENFJ", "INFJ", "INTJ", "ENTJ"
]
# alphabetically sorted
VALID_MBTI_TYPES = sorted(VALID_MBTI_TYPES)


def make_mbti_reddit_plot():
    """
    Plot the distribution of the guesses on the original MBTI data
    """
    fname = "mbti_reddit.jsonl"
    og_guesses = []
    with open(fname, "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            og_guesses.append(data["commented_types"])
    guess_counts = {}
    for guess in og_guesses:
        for type, score in guess.items():
            if type not in guess_counts:
                guess_counts[type] = 0
            guess_counts[type] += score
    # clear the plot
    plt.clf()
    # bar plot of total number of guesses per type
    plt.bar(guess_counts.keys(), guess_counts.values())
    plt.title("Total number of guesses per type for original Reddit data")
    plt.xlabel("Type")
    plt.xticks(rotation=45)
    plt.ylabel("Number of guesses")
    # save the plot
    plt.savefig("plots/total_guesses_mbti_reddit.png")


def make_og_submission_guess_list(llm_guess_fnames):
    og_file = "mbti_reddit.jsonl"
    og_guesses = []
    with open(og_file, "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            og_guesses.append(data["commented_types"])
    llm_guesses = {}
    guess_counts = {}
    for llm_guess_fname in llm_guess_fnames:
        model_name = os.path.basename(llm_guess_fname).split("_")[1]
        model_suffix = os.path.basename(
            llm_guess_fname).split("_")[-1].split(".")[0]
        model_name = f"{model_name}_{model_suffix}"
        llm_guesses[model_name] = {}
        guess_counts[model_name] = {}
        with open(llm_guess_fname, "r") as f:
            for i, line in enumerate(f.readlines()):
                data = json.loads(line)
                guess = data["clean_guess"]
                if guess not in VALID_MBTI_TYPES:
                    continue
                if guess not in llm_guesses[model_name]:
                    llm_guesses[model_name][guess] = {}
                if guess not in guess_counts[model_name]:
                    guess_counts[model_name][guess] = 0
                guess_counts[model_name][guess] += 1
                og_guess = og_guesses[i]
                for type, score in og_guess.items():
                    if type not in llm_guesses[model_name][guess]:
                        llm_guesses[model_name][guess][type] = 0
                    llm_guesses[model_name][guess][type] += score
    return llm_guesses, guess_counts


def logit_info(fname):
    """
    Get the average top logit value and the count of refusals
    """
    top_log_prob = []
    refusals = 0
    total_len = 0
    with open(fname, "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            total_len += 1
            if "The" in data["llm_guess"] and "zodiac" in fname:
                refusals += 1
            else:
                logprobs = data["logprobs"]
                first_token_logprob_key = next(iter(logprobs))
                top_logit = logprobs[first_token_logprob_key][0]["logprob"]
                top_log_prob.append(math.exp(top_logit))
    avg_top_logit = sum(top_log_prob) / len(top_log_prob)
    refusal_rate = refusals / total_len
    return avg_top_logit, refusal_rate


def make_seaborn_plot(llm_guesses, guess_counts):
    for model_name, guess_dict in llm_guesses.items():
        # alphabetical order
        guess_dict = dict(sorted(guess_dict.items()))
        # rotate the x-axis labels for better readability
        # clear the plot
        plt.clf()
        # heatmap of accuracy per type
        sns.heatmap([[guess_dict[guess].get(type, 0) / guess_counts[model_name][guess] for type in VALID_MBTI_TYPES]
                     for guess in guess_dict.keys()], xticklabels=VALID_MBTI_TYPES, yticklabels=guess_dict.keys())
        plt.title(f"Accuracy per type for {model_name}")
        plt.xlabel("Type")
        plt.xticks(rotation=45)
        plt.ylabel("Guess")
        # save the plot
        plt.savefig(f"plots/accuracy_{model_name}.png")


if __name__ == "__main__":
    # plot the distribution of the guesses on the original MBTI data
    make_mbti_reddit_plot()
    fnames = ["llm_guesses/mbti_gpt-4_brief.jsonl", "llm_guesses/mbti_gpt-3.5-turbo_brief.jsonl", "llm_guesses/mbti_claude-3-opus-20240229_brief.jsonl",
              "llm_guesses/mbti_claude-3-haiku-20240307_brief.jsonl", "llm_guesses/mbti_claude-3-haiku-20240307_long.jsonl"]
    llm_guesses, guess_counts = make_og_submission_guess_list(fnames)
    for fname in ["llm_guesses/mbti_gpt-4_brief.jsonl", "llm_guesses/mbti_gpt-3.5-turbo_brief.jsonl", "llm_guesses/zodiac_gpt-4_brief_manylogs.jsonl", "llm_guesses/zodiac_gpt-3.5-turbo_brief_manylogs.jsonl"]:
        avg_top_logit, refusals = logit_info(fname)
        print("Logprob for ", fname, ":",
              avg_top_logit, "Refusal rate:", refusals)
    for model_name, guess_dict in llm_guesses.items():
        # alphabetical order
        guess_dict = dict(sorted(guess_dict.items()))
        # rotate the x-axis labels for better readability
        # clear the plot
        plt.clf()
        # bar plot of total number of guesses per type
        plt.bar(guess_dict.keys(), guess_counts[model_name].values())
        plt.title(f"Total number of guesses per type for {model_name}")
        plt.xlabel("Type")
        plt.xticks(rotation=45)
        plt.ylabel("Number of guesses")
        # save the plot
        plt.savefig(f"plots/total_guesses_{model_name}.png")

        # clear the plot
        plt.clf()
    # seaborn plot of accuracy per type
    make_seaborn_plot(llm_guesses, guess_counts)
