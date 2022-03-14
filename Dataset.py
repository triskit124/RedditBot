import torch
import pickle
import os

from RedditScraper import RedditScraper


class Dataset(torch.utils.data.Dataset):
    """
    Dataset is the Class to store and retrieve Training/Validation/Test data for the RedditNN Class.
    Inherits from the typical PyTorch Dataset Class
    """

    def __init__(self, args):
        self.sequenceLength = args.sequence_length          # (int) length of input into NN
        self.posts = None                                   # (List)
        self.bodyIndexed = []
        self.titleIndexed = []
        self.uniqueTitleWords = set()
        self.uniqueBodyWords = set()
        self.uniqueWords = set()
        self.wordToIdx = {}
        self.idxToWord = {}

    def __len__(self):
        return len(self.bodyIndexed) - self.sequenceLength

    def __getitem__(self, index):
        return (
            torch.tensor(self.bodyIndexed[index:index + self.sequenceLength]),
            torch.tensor(self.bodyIndexed[index + 1:index + self.sequenceLength + 1])
        )

    def scrapeNewPostsToFile(self, sub, numPosts=1, filename='tmp.pkl', appendFile=True):
        """
        Scrapes a new dataset from Reddit by pulling random posts from a specified subreddit.
        :param sub: (String) subreddit to scrape posts from, Ex: "CasualConversation"
        :param numPosts: (int) number of posts to scrape
        :param filename: (String) path of file to Pickle to
        :param appendFile: (Bool) If False, will overwrite contents of file
        :return: Saves dataset file of the form: List[Tuple(int, List[String], List[String])]
        """

        # post scraping is facilitated via the Class: RedditScraper(), see RedditScraper.py
        # RedditScraper.getPostsFromSub() returns a pandas dataframe of posts
        redditScraper = RedditScraper()
        posts = redditScraper.getPostsFromSub(sub, numPosts)

        # open pre-existing file if wish to append file
        if appendFile and os.path.getsize(filename) > 0:
            with open(filename, 'rb') as f:
                contents = pickle.load(f)
                offset = len(contents)
        else:
            contents = []
            offset = 0

        # append contents for each post that has been scraped
        for i in range(len(posts)):
            post = posts.iloc[i]
            contents.append((i + offset, post['title'].replace('\n', '').split(" "), post['body'].replace('\n', '').split(" ")))

        # save contents to a binary Pickled file
        with open(filename, 'wb') as f:
            pickle.dump(contents, f)
            print("\tSaved %2f posts from %s to file %s" % (numPosts, sub, filename))

    def loadPostsFromFile(self, filename='tmp.pkl'):
        """Loads a dataset of posts from a Pickled file.

        :param filename: path to the Pickeld file to load
        :return: Modifies the following instance variables:
                - self.posts: (List[Tuple(int, List[String], List[String])]) dataset of posts
                - self.uniqueTitleWords: (Set) All unique words observed in dataset of post titles
                - self.uniqueBodyWords: (Set) All unique words observed in dataset of post bodies
                - self.uniqueWords: (Set) All unique words observed in titles/posts
                - self.wordToIdx: (Dict) Maps unique words to unique integers
                - self.idxToWord: (Dict) Maps unique integers to unique words
                - self.bodyIndexed: (List) Concatenated indexed bodies of posts
                - self.titleIndexed: (List) Concatenated indexed titles of posts
        """

        with open(filename, 'rb') as f:
            posts = pickle.load(f)
            #print(posts)

        # TODO: clean this in the scaper
        cleaned_posts = []
        for i, post in enumerate(posts):
            post1 = [w.replace(".", " . ").replace(",", " , ").replace(")", " ) ").replace("(", " ( ").replace("?", " ? ").replace("!", " ! ").lower() for w in post[1]]
            post2 = [w.replace(".", " . ").replace(",", " , ").replace(")", " ) ").replace("(", " ( ").replace("?", " ? ").replace("!", " ! ").lower() for w in post[2]]

            post1List = ""
            post2List = ""
            for w in post1:
                post1List += w + " "

            for w in post2:
                post2List += w + " "

            post1List = post1List.split()
            post2List = post2List.split()

            cleaned_posts.append((i, post1List, post2List))

        #print(cleaned_posts)

        self.posts = cleaned_posts

        # loop through posts and find unique words that occur in both title and body
        for post in cleaned_posts:
            self.uniqueTitleWords.update(set(post[1]))
            self.uniqueBodyWords.update(set(post[2]))
            self.uniqueWords = self.uniqueTitleWords | self.uniqueBodyWords # set concatenation operation

        # generate a unique (integer) index for each unique word in the dataset
        self.wordToIdx = {word: index for index, word in enumerate(self.uniqueWords)}
        self.idxToWord = {index: word for index, word in enumerate(self.uniqueWords)}

        # loop through each post and replace words with their hashed indices
        for i, post in enumerate(cleaned_posts):
            _titleIndexed = [self.wordToIdx[w] for w in post[1]]
            _bodyIndexed = [self.wordToIdx[w] for w in post[2]]
            self.bodyIndexed.extend(_bodyIndexed)
            self.titleIndexed.extend(_titleIndexed)

        #print(self.titleIndexed)
        #print(self.bodyIndexed)
        #print(self.wordToIdx)
        #print(len(self.uniqueWords))

