def Flatten(t):
	return [item for sublist in t for item in sublist]

def IterateBuildTree(tree_format, prearmed, symbol):
    if len(prearmed) == 0:
        return tree_format,prearmed
    if len(prearmed) == 1:
        tree_format = prearmed
        prearmed = []
        return tree_format,prearmed
    else:
        if len(tree_format) == 0:
            aux = [prearmed[1],[prearmed[0]], [prearmed[2]]]
            prearmed = prearmed[3::]
        else:
            aux = [prearmed[0],tree_format,prearmed[1]]
            prearmed = prearmed[2::]
    return IterateBuildTree(aux, prearmed, symbol)


def InnerJoinsIntraBGPS(bgp, symbol):
    prearmed = []
    tree_format = []
    predicates = []
    for k in range(len(bgp)):
        if k == 0:
            if 'NONE' not in bgp[k]['P']:
                predicates.append(bgp[k]['P'])
            prearmed.append(bgp[k]['triple_type'] + symbol + bgp[k]['P'])
        else:
            if 'NONE' not in bgp[k]['P']:
                predicates.append(bgp[k]['P'])
            prearmed.append('JOIN' + symbol + symbol.join(predicates[:k+1]))
            prearmed.append(bgp[k]['triple_type'] + symbol + bgp[k]['P'])
    tree_format, prearmed = IterateBuildTree(tree_format, prearmed, symbol)
    return tree_format, predicates


def IterateBuildTreeBetweenBGPS(tree_format, prearmed, symbol):
    if len(prearmed) == 0:
        return tree_format,prearmed
    if len(prearmed) == 1:
        tree_format = prearmed
        prearmed = []
        return tree_format,prearmed
    else:
        if len(tree_format) == 0:
            aux = [prearmed[1],prearmed[0], prearmed[2]]
            prearmed = prearmed[3::]
        else:
            aux = [prearmed[0],tree_format,prearmed[1]]
            prearmed = prearmed[2::]
    return IterateBuildTreeBetweenBGPS(aux, prearmed, symbol)



def TreeFormat(new_dicto, symbol):
    bgp_joins = []
    bgp_type = []
    tree_format = []
    prearmed = []
    list_current_predicates = []
    for k, v in new_dicto.items():
        bgp, current_predicates = InnerJoinsIntraBGPS(v['bgp_list'], symbol)
        type_opt = v['opt']
        bgp_joins.append(bgp)
        bgp_type.append(type_opt)
        list_current_predicates.append(current_predicates)

    for k in range(len(bgp_joins)):
        if k == 0:
            prearmed.append(bgp_joins[k])
        else:
            if bgp_type[k] == 0:
                prearmed.append('JOIN')
                prearmed.append(bgp_joins[k])
            if bgp_type[k] == 1:
                prearmed.append('LEFT_JOIN')
                prearmed.append(bgp_joins[k])
    if len(new_dicto.keys()) == 1:
        tree_format = prearmed[0]
    else:
        tree_format, prearmed = IterateBuildTreeBetweenBGPS(tree_format, prearmed, symbol)

    return tree_format

def TreeFormat_all(new_dicto, symbol):
    bgp_joins = []
    bgp_type = []
    tree_format = []
    prearmed = []
    list_current_predicates = []
    for k, v in new_dicto.items():
        bgp, current_predicates = InnerJoinsIntraBGPS(v['bgp_list'], symbol)
        type_opt = v['opt']
        bgp_joins.append(bgp)
        bgp_type.append(type_opt)
        list_current_predicates.append(current_predicates)

    for k in range(len(bgp_joins)):
        if k == 0:
            prearmed.append(bgp_joins[k])
        else:
            if bgp_type[k] == 0:
                prearmed.append('JOIN' + symbol + symbol.join(Flatten(list_current_predicates[:k+1])))
                prearmed.append(bgp_joins[k])
            if bgp_type[k] == 1:
                prearmed.append('LEFT_JOIN' + symbol + symbol.join(Flatten(list_current_predicates[:k+1])))
                prearmed.append(bgp_joins[k])
    if len(new_dicto.keys()) == 1:
        tree_format = prearmed[0]
    else:
        tree_format, prearmed = IterateBuildTreeBetweenBGPS(tree_format, prearmed, symbol)

    return tree_format