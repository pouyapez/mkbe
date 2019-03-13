from nltk.translate.bleu_score import corpus_bleu


def list_of_s(file, type = "r"):
	data = []
	list = []
    	with open(file) as f: #open(p)
		data = f.readlines() + data

	for line in data:
		if type == "r":
			list += [[line.split()]]
		if type == "h":
			list += [line.split()]
	
	return list


def Bleu_S(ref, hypo):
	score = corpus_bleu(ref, hypo) ##(ref, hypo) or (ref, hypo, weights=(0.5, 0.5))
	return score

if __name__ == "__main__":
	ref = "snli_lm/test.txt"##"output/example/end_of_epoch13_lm_generations.txt"###"snli_lm/test.txt"
	
	for i in range(44):
		i+= 6
		hypo = "output/example/end_of_epoch"+str(i)+"_lm_generations.txt"

		ref_list = list_of_s(ref)
		hypo_list = list_of_s(hypo, type = "h")

		#print len(ref_list), len(hypo_list)

		bleu_score = Bleu_S(ref_list, hypo_list)
		print bleu_score
		