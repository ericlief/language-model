SHELL=/bin/bash


# Makefile for NLP Tech HW #1
default: TEXTEN1.txt TEXTCZ1.txt exec entropy stats messup smooth
# entropy: download download2 set_up
# messup: download download2 set_up

# Download data
TEXTEN1.txt:
	wget http://ufal.mff.cuni.cz/~hajic/courses/npfl067/TEXTEN1.txt

TEXTCZ1.txt:
	wget http://ufal.mff.cuni.cz/~hajic/courses/npfl067/TEXTCZ1.txt

exec : entropy.py languageModel.py
	chmod a+x main.py

# Compute conditional entropy texts
entropy: exec TEXTEN1.txt TEXTCZ1.txt
	cat TEXTEN1.txt | ./main.py --entropy ent_en.txt
	cat TEXTCZ1.txt | ./main.py  --entropy ent_cz.txt

# Messup/randomize text with x likelihood
messup: exec TEXTEN1.txt TEXTCZ1.txt
	cat TEXTEN1.txt | ./main.py --randomchars chars_en.txt
	cat TEXTCZ1.txt | ./main.py --randomchars chars_cz.txt
	cat TEXTEN1.txt | ./main.py --randomwords words_en.txt
	cat TEXTCZ1.txt | ./main.py --randomwords words_cz.txt

smooth: exec TEXTEN1.txt TEXTCZ1.txt
	cat TEXTCZ1.txt | ./main.py --smooth error.txt			# to show error of smoothing on training data
	cat TEXTEN1.txt | ./main.py --smooth error.txt			# to show error of smoothing on training data
	cat TEXTEN1.txt | ./main.py --smooth smooth_en.txt
	cat TEXTCZ1.txt | ./main.py --smooth smooth_cz.txt

stats: exec TEXTEN1.txt TEXTCZ1.txt
	cat TEXTEN1.txt | ./main.py --stats stats_en.txt
	cat TEXTCZ1.txt | ./main.py --stats stats_cz.txt

clean:
	rm TEXT*

unittest:
	python  -m unittest testLanguageModel.py
	python  -m unittest testEntropy.py
	python  -m unittest testSmoothedLM.py



