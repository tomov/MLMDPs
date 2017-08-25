fprintf('\n\n\n----------  LMDP ------------\n\n\n');


L = LMDP([
    '$####';
    '.#X..';
    '.....']);
L.solveLMDP();
L.sample();


fprintf('\n\n\n--------- MLMDP -----------\n\n\n');


M = MLMDP([
    '0####';
    '.#X..';
    '0...0']);
M.presolve();

M.solveMLMDP([10 -1 -1]');
M.sample();

M.solveMLMDP([-1 -1 10]');
M.sample();


fprintf('\n\n\n--------- AMLMDP -----------\n\n\n');


A = AMLMDP([
    '0####';
    '.#X..';
    '0...0'], [2 5 6]);
A.presolve();

A.solveMLMDP([10 -1 -1 -1 -1 -1]');
A.sample();

A.solveMLMDP([-1 -1 -1 -1 -1 10]');
A.sample();

%%
%

map = [
    '$####';
    'S#X..';
    '0.S.0'];
H = HMLMDP(map);