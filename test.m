fprintf('\n\n\n----------  LMDP ------------\n\n\n');


L = LMDP([
    '$####';
    '.#X..';
    '.....']);
L.solveLMDP();
L.sample();


%% MLMDP
%
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


%% augmented MLMDP
%
fprintf('\n\n\n--------- AMLMDP -----------\n\n\n');


A = AMLMDP([
    '0####';
    '.#X..';
    '0...0'], [2 6]);
A.presolve();

A.solveMLMDP([10 -1 -1 -1 -1]');
A.sample();

A.solveMLMDP([-1 -1 -1 -1 10]');
A.sample();


%% HMLMDP
%

fprintf('\n\n\n\n\n\n\n\n--------------------------------- HMLMDP ----------------------------------\n\n\n\n\n\n\n');

map = [
    '$####';
    'S#X..';
    '0.S.0'];
H = HMLMDP(map);
H.solve();



%% HMLMDP
%

fprintf('\n\n\n\n\n\n\n\n--------------------------------- HMLMDP ----------------------------------\n\n\n\n\n\n\n');

map = [
    '$####';
    'S#X..';
    '0.S.0'];
H = HMLMDP(map);
H.solve();

%% Big HMLMDP
%

fprintf('\n\n\n\n\n\n\n\n--------------------------------- big HMLMDP ----------------------------------\n\n\n\n\n\n\n');

map = [
    'SX...#.....';
    '.....#.....';
    '..#S.......';
    '.#...#.S##.';
    '.....#...0.';
    '#.####.....';
    '.....###.##';
    '..#..#...S.';
    '..#..#.....';
    '.0S....S#..';
    '.....#..$..'];
H = HMLMDP(map);
H.solve();

self = H; % for debugging


