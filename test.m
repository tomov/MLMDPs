L = LMDP([
    '$####';
    '.#X..';
    '.....']);
L.solveLMDP();
L.sample();


fprintf('\n\n\n---------------\n\n\n');


M = MLMDP([
    '0####';
    '.#X..';
    '0...0']);
M.presolve();

M.solveMLMDP([10 -1 -1]');
M.sample();

M.solveMLMDP([-1 -1 10]');
M.sample();