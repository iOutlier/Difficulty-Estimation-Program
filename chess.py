#!/usr/bin/python

# library for reading the chesstempo comma separated value data
import csv
# library for opening the chess engine, in our case Rybka
import subprocess
# library for matching the output from the chess engine
import re
# library for working with max system numbers
import sys

# library for constructing a graph of sensible moves
import networkx as nx
# library for drawing the graph
import matplotlib.pyplot as plt

# library for representing our chess games
from Chessnut import Game

# dictionary for tracing the distance per level
accumulated_distance = {}

# library for chess indices
indices = {
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4,
    'e': 5,
    'f': 6,
    'g': 7,
    'h': 8,
}

# library for the values of chess pieces
values = {
    'p': 1,
    'P': 1,
    'n': 3,
    'N': 3,
    'b': 3,
    'B': 3,
    'r': 5,
    'R': 5,
    'q': 9,
    'Q': 9,
    'k': 0,
    'K': 0,
}

class FenTree():
    # Class to represend each node (which is a tree itself) in the tree of chess moves

    # ID of the node, so that we can differentiate between the similar moves when we are drawing them
    id = 0
    
    def __init__(self, root, last_move, score, depth=0):
        self.fen = root
        self.moves = []
        self.last_move = last_move
        self.depth = depth
        self.score = score
        self.id = FenTree.id
        FenTree.id += 1

    def __str__(self):
        return "{0}. {1},{2}".format(self.id, self.last_move, self.score)

    def __unicode__(self):
        return unicode("{0}. {1},{2}".format(self.id, self.last_move, self.score))

    def add_move(self, new_fen, move, score):
        self.moves.append(FenTree(root=new_fen, last_move=move, depth=self.depth + 1, score=score))

    def display_the_tree(self, display_level=True, level=0):
        print '\t' * level, self.last_move, ", score:", abs(int(self.score)), "level:", level
        for move in self.moves:
            if self.moves.index(move) == 0:
                display_level = True
            else:
                display_level = False
            move.display_the_tree(display_level, level + 1)


def display_the_size(chess_tree, size, level=0, max_level=4):
    if level >= max_level:
        return
    if size.get(level):
        "level 1: len(moves[0]) + len(moves[1]) + ..."
        size[level] += len(chess_tree.moves)
    else:
        "level 0: len(moves)"
        size[level] = len(chess_tree.moves)

    for move in chess_tree.moves:
        display_the_size(move, size, level + 1)

def add_children(G, father):
    for child in father.moves:
        G.add_node(str(child))
        G.add_edge(str(father), str(child))
        add_children(G, child)

def draw_graph(tree, problem_id):
    root = tree

    G = nx.DiGraph()
    G.add_node(str(root))

    add_children(G, root)

    nx.write_dot(G, '%s.dot' % problem_id)

    plt.title("%s" % root.fen)
    pos = nx.graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=False, arrows=False)
    plt.savefig('arbitrary_%s.png' % problem_id)
       

def build_the_tree(tree_of_moves, reasonable_moves, chess_game, rybka, first_player_color, max_tree_depth=3):
    if tree_of_moves.depth > max_tree_depth:
        return
    default_fen = str(chess_game)
    i = 0

    for line in reasonable_moves:
        move = line.split()
        #print "tree:", tree_of_moves.fen
        #print "move[15]:", move[15],
        #print "type:", move[6],
        #print "score:", move[7]

        chess_game.apply_move(move[15])
        tree_of_moves.add_move(str(chess_game), move[15], move[7])

        if not (move[6] == 'mate' and str(move[7]) == '1'):
            start_new_game(rybka, chess_game)
            #print 'fisrt_player_color', first_player_color
            #print 'current player to move', str(chess_game).split()[1]
            if str(chess_game).split()[1] == first_player_color:
                new_cp_parameter = 200
            else:
                new_cp_parameter = 50
            #print "for " + str(chess_game).split()[1] + ", the cp parameter is " + str(new_cp_parameter)
            reasonable_moves = find_all_reasonable_moves(rybka, current_depth=tree_of_moves.depth+1, cp_parameter=new_cp_parameter)

            if reasonable_moves:    # If there are any reasonable moves
                build_the_tree(tree_of_moves.moves[i], reasonable_moves, chess_game, rybka, first_player_color, max_tree_depth)

        # reset the FEN
        chess_game.reset(fen=default_fen)

        # next FEN
        i += 1

def calculate_values(color, fen):
    placement = fen.split()[0]
    total_value = 1
    if color == 'w':
        for piece in placement:
            if piece.isupper():
                total_value += values[piece]
    elif color == 'b':
        for piece in placement:
            if piece.islower():
                total_value += values[piece]
    # print color, total_value
    return total_value

def add_the_tree(chess_tree, problem_id, fen, prev_move, blitz_rating, std_rating):
    """
    " Tree size, for each level
    """
    global accumulated_distance
    size = {}

    display_the_size(chess_tree, size)

    print size

    nodes = sorted(size.items(), key=lambda y: y[0])

    fathers = [0] * len(nodes)
    sons = [0] * len(nodes)
    branching_factor = {}

    i = 0
    for x in range(len(nodes) - 1):
        fathers[i] += nodes[x][1]
        i += 1
    i = 0
    for x in range(1, len(nodes)):
        sons[i] += nodes[x][1]
        i += 1
    for x in range(len(nodes)):
        if float(fathers[x]) > 0:
            branching_factor[x] = sons[x] / float(fathers[x])
        else:
            branching_factor[x] = 0

    position = fen.split()[0]
    figures = re.findall('[a-zA-Z]', position)
    types_of_pieces = len(set(figures))
    number_of_pieces = len(figures)

    """
    " 13.10.2014
    " Calculate the piece values
    " example: rnbqkbnr would yield 5 + 3 + 3 + 9 + 0 + 3 + 3 + 5 = 31 points for black
    " RNBQKB1R would yield 5 + 3 + 3 + 9 + 0 + 3 + 5 = 28 points for white
    " If it's white to move, the ratio will then be 28 / 31 = 0.9032258
    " The ratio then helps determine the difficulty of the game
    """
    new_fen = Game(fen=fen)
    new_fen.apply_move(prev_move)
    piece_value_w = calculate_values('w', str(new_fen))
    piece_value_b = calculate_values('b', str(new_fen))
    next_to_move = str(new_fen).split()[1]
    if next_to_move == 'w':
        piece_value_ratio = piece_value_w / float(piece_value_b)
    else:
        piece_value_ratio = piece_value_b / float(piece_value_w)
    # End of piece value calculation code

    with open('C:\Users\Simon\Dropbox\FRI\Diplomsko delo\Program\processed_data.csv', 'a+b') as csv_file:
        fen_writer = csv.writer(csv_file, delimiter=',')

        my_buffer = [problem_id, fen, prev_move, blitz_rating, std_rating]

        for i in range(4):
            my_buffer.append(size.get(i, 0))
        average_branching_factor = 0
        for i in range(4):
            my_buffer.append(branching_factor.get(i, 0))
            average_branching_factor += branching_factor.get(i, 0)
        average_branching_factor /= 4.0
        my_buffer.append(average_branching_factor)
        my_buffer.append(number_of_pieces)
        my_buffer.append(types_of_pieces)
        for i in range(4):
            my_buffer.append(accumulated_distance.get(i, 0))
        my_buffer.append(piece_value_ratio)

        fen_writer.writerow(my_buffer)

        # Reset the distance calculator for the moves
        accumulated_distance = {}


def calculate_distance(move):
    start_square = move[:2]
    end_square = move[2:4]
    x1 = indices.get(start_square[0])
    x2 = indices.get(end_square[0])
    # print x1, x2
    y1 = int(start_square[1])
    y2 = int(end_square[1])
    # print y1, y2

    distance = abs(x1 - x2) + abs(y1 - y2)

    return distance


def separate_the_reasonable_ones(list_of_moves, current_depth, cp_parameter=50):
    """
    " Find the moves with the appropriate cp or mate
    """
    global accumulated_distance
    list_of_reasonable_moves = []
    no_reasonable_moves = True
    best_cp = -sys.maxint - 1

    # Most important part of the program
    for line in reversed(list_of_moves):  # Reversed list, so we can start at max CP
        move = line.split()

        if move[6] == 'cp':  # If the value is measured in CP (not mate)
            if int(move[7]) >= cp_parameter:  # if 553 > 200:
                list_of_reasonable_moves.append(line) # the whole line is appended, and split afterwards
                no_reasonable_moves = False # We just found a reasonable move, so we don't need any unreasonable ones
            elif no_reasonable_moves and (len(list_of_reasonable_moves) <= 3):
                """ If the move doesn't match the cp requirement
                " take the moves that will save you the most
                " and aren't more than 50 cp apart from each other
                " and we can only take 3 of those
                """
                if int(move[7]) > best_cp:
                    best_cp = int(move[7])  # the first unreasonable move will be the best_cp argument
                if abs(abs(int(move[7])) - abs(best_cp)) <= 50:  # if abs(abs(-103) - abs(-53)) <= 50
                    list_of_reasonable_moves.append(line)   # again, the whole line is appended

        elif move[6] == 'mate':     # If the value is measured in MATE (not cp)
            # If we can win
            if int(move[7]) > 0:
                list_of_reasonable_moves.append(line)
                no_reasonable_moves = False
            # if we can't move, but there isn't any other option left
            elif no_reasonable_moves and (len(list_of_reasonable_moves) <= 3):
                list_of_reasonable_moves.append(line)

        elif move[6] == 'lowerbound':
            if int(move[7]) >= cp_parameter:
                list_of_reasonable_moves.append(line)
            print 'found lower bound'

        elif move[6] == 'upperbound':
            if int(move[7]) >= cp_parameter:
                list_of_reasonable_moves.append(line)
            print 'found upper bound'

    for move in list_of_reasonable_moves:
        accumulated_distance[current_depth] = accumulated_distance.get(current_depth, 0) + calculate_distance(move.split()[15])

    return list_of_reasonable_moves


def find_all_at_max_depth(all_moves):
    list_of_moves = []  # at the max depth

    # start from the back, and append until you get to MultiPV == 1
    for rybka_line in reversed(all_moves):
        list_of_moves.append(rybka_line)
        if int(rybka_line.split()[2]) == 1:   # If MultiPV == 1, we got all the data we need
            break
    return list_of_moves


def get_all_moves(rybka):
    output = []  # list of all the moves
    # Until we get to the ^bestline xxxx ponder xxxx$ line, don't stop reading
    last_line = re.compile('^bestmove')

    dummy_line = rybka.stdout.readline()

    while True:
        rybka_line = rybka.stdout.readline()
        if last_line.match(rybka_line):
            break
        if len(rybka_line.split()) > 4 and int(rybka_line.split()[4]) == 8:
            output.append(rybka_line)

    return output


def find_all_reasonable_moves(rybka, current_depth, search_depth=8, cp_parameter=50):

    all_moves = get_all_moves(rybka)

    list_at_given_depth = find_all_at_max_depth(all_moves)
    
    reasonable_moves = separate_the_reasonable_ones(list_at_given_depth, current_depth, cp_parameter=cp_parameter)

    return reasonable_moves


def start_new_game(rybka_connection, chess_game, search_depth=8):
    rybka_connection.stdin.write('ucinewgame\n')  # Start a new game
    rybka_connection.stdin.write('isready\n')  # Wait for Rybka to get ready
    rybka_line = wait_for_response(rybka_connection, 'readyok\r\n')     # Until Rybka is ready

    rybka_connection.stdin.write('setoption name Hash value 128\n')  # 128 MB Hash table memory
    # houdini_line = wait_for_response(houdini_connection, 'info string 256 MB Hash\r\n')
    rybka_connection.stdin.write('setoption name Clear Hash\n')  # Clearing the Hash enables newgame
    rybka_line = wait_for_response(rybka_connection, 'info string hash cleared\r\n')
    rybka_connection.stdin.write('setoption name MultiPV value 100\n')
    rybka_connection.stdin.write('isready\n')  # Wait for Rybka to get ready
    rybka_line = wait_for_response(rybka_connection, 'readyok\r\n')     # Until Rybka is ready
    
    rybka_connection.stdin.write('position fen ' + str(chess_game) + '\r\n')  # Set the position in FEN notation
    rybka_connection.stdin.write('go depth ' + str(search_depth) + '\r\n')  # 8 is enough, cause of the Quiescence search


def play_one_game(fen, last_move, rybka, max_tree_depth=3):
    chess_game = Game(fen=fen)
    chess_game.apply_move(last_move)

    print chess_game

    start_new_game(rybka, chess_game, search_depth=8)
    # print "for " + str(chess_game).split()[1] + ", the cp parameter is " + str(200)
    reasonable_moves = find_all_reasonable_moves(rybka, current_depth=0, search_depth=8, cp_parameter=200)   # CP is 200 the first time

    tree_of_moves = FenTree(root=fen, last_move=last_move, score=0)

    first_player_color = str(chess_game).split()[1]
    # print str(chess_game).split()[1]

    if reasonable_moves:  # If there are any reasonable_moves
        build_the_tree(tree_of_moves, reasonable_moves, chess_game, rybka, first_player_color, max_tree_depth)

    #[x.display_the_tree() for x in tree_of_moves.moves]

    return tree_of_moves


def quit_rybka(process):
    """
    " Quit the Rybka.exe program/process
    """
    try:
        process.kill()
    except OSError:
        print "OSError: Cannot close Rybka.exe"


def wait_for_response(rybka, expected_response):
    line = ''
    while True:
        line = rybka.stdout.readline()
        if line == expected_response:
            break
    return line


def establish_connection():
    """
    " In order to communicate with the Rybka chess engine, we need to establish a connection
    """
    try:
        rybka = subprocess.Popen(["C:\Users\Simon\Dropbox\FRI\Diplomsko delo\Program\Rybka.exe"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        # Rybka doesn't have any headers
        # houdini_line = wait_for_response(houdini, 'info string 128 MB Hash\r\n')

        rybka.stdin.write('uci\n')  # Tell the engine that we will use the UCI protocol

        # Wait for Rybka to prepare for the protocol
        rybka_line = wait_for_response(rybka, 'uciok\r\n')
    except OSError:
        print "OSError: Cannot open Rybka.exe"
        sys.exit()  # Just exit the program
    return rybka


def read_the_csv(datafile_name):
    chess_tempo_data = []
    with open(datafile_name, 'rb') as csv_file:
        data_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
        for row in data_reader:
            chess_tempo_data.append(row)
    return chess_tempo_data


def main_program(datafile_name="C:\Users\Simon\Dropbox\FRI\Diplomsko delo\Program\chesstempo.csv", number_of_positions=-1):
    """
    " Read the data from the .csv file (chesstempo.csv), line by line (each line is one chess game)
    " and for given number of positions search the possible moves and their counter-moves,
    " than make a tree from it, and save it in the output .csv (fens.csv)
    """
    chess_tempo_data = read_the_csv(datafile_name)

    rybka_program = establish_connection()    # with Rybka.exe

    if number_of_positions == -1:
        # If this parameter isn't given
        # just process all the data (every position)
        number_of_positions = len(chess_tempo_data)

    # The index in the next line should be 1:number_of_positions
    # to process the whole data
    for chess_game in chess_tempo_data[4:5]:
        # Send the position in the FEN notation and the last move
        chess_tree = play_one_game(chess_game[1], chess_game[2], rybka_program)
        #draw_graph(chess_tree, chess_game[0])
        # Reset the object counter for the tree nodes
        FenTree.id = 0
        # problem_id, fen, prev_move, blitz_rating, std_rating
        add_the_tree(chess_tree, chess_game[0], chess_game[1], chess_game[2], chess_game[3], chess_game[4])

    quit_rybka(rybka_program)
