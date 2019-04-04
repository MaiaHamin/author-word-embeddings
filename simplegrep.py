def reversex(regex, string):
    if (match(regex, string)):
        return regex
    else:
        return longest_pattern(regex, string)

def longest_pattern(regex, string):
    if (len(string) == 0):
        return ""
    string_len = len(string)
    regex_len = len(regex)
    max_len = 0
    reg_max = 0
    string_max = 0
    for i in range(regex_len):
        for j in range(string_len):
            if (regex[i] == string[j]):
                k = 0
                while (i + k < regex_len and
                j + k < string_len and
                regex[i + k] == string[j + k]):
                    k += 1
                if (max_len < k):
                    max_len = k
                    reg_max = i
                    string_max = j
        if (max_len == 0):
            return "*"
    frontreg = longest_pattern(regex[:reg_max], string[:string_max])
    backreg = longest_pattern(regex[reg_max + max_len:], string[string_max + max_len:])
    return frontreg + regex[reg_max: reg_max + max_len] + backreg


def fuzzy_grep(regex, string, thresh):
    r_len = len(regex)
    s_len = len(string)
    matches = []
    for i in range(s_len):
        i_score = 0
        for j in range(min(r_len, s_len - i)):
            if not (match(regex[j], string[i + j])):
                i_score += 1
        if (i_score < thresh):
            matches.append(string[i: i + r_len])
    return matches

match(regexp, text):
	if (regexp[0] == '^')
		return matchhere(regexp+1, text)
	while (len(text) != 0):
		return matchhere(regexp, text[1:])
	} ;
	return False;
}

matches = fuzzy_grep("* and *", "peas and carrots and sandy beaches", 2)
for m in matches:
    print(m)
