% Multitask MLMDP as described in Saxe et al (2017)
%
classdef MLMDP < LMDP

    properties (Constant = true)
        R_B_goal = 0; % reward for goal boundary state for the given task
        R_B_nongoal = -Inf; % reward for the other boundary states
    end

    properties (Access = public)
        qi = []; % exponentiated rewards for internal states only
        Qb = []; % exponentiated rewards for boundary states (row) for each task (col)
        Zi = []; % desirability function for all states (row) each task (col)
    end

    methods 

        % Create a MLMDP from a maze with multiple goal states;
        % creates a separate task for each goal state
        %
        function self = MLMDP(map)
            self = self@LMDP(map);
            
            Nb = numel(self.B);

            self.qi = self.q(self.I); % internal states have the same rewards for each task

            % Qb = Nb x Nb identity matrix
            % => each column is a task with 1 goal state = the corresponding boundary state
            % and has the exponentiated rewards for all boundary states (as rows)
            %
            Rb = ones(Nb, Nb) * self.R_B_nongoal;
            Rb(logical(eye(Nb))) = self.R_B_goal;
            Qb = exp(Rb / self.lambda);
            self.Qb = Qb;

            self.q = []; % doesn't make sense any more 
            self.sanityMLMDP();
        end

        % 'Pre-solve' an initialized MLMDP for all basis tasks
        % => compute desirability matrix Z
        %
        function presolve(self)
            Zi = [];
            a = [];
            for i = 1:size(self.Qb, 2) % for each subtask

                % set rewards according to subtask
                %
                qb = self.Qb(:,i); % subtask i
                self.q = [self.qi; qb];

                % call regular LMDP solver
                %
                self.solveLMDP();
                
                if isempty(Zi)
                    Zi = self.z(self.I);
                else
                    Zi = [Zi, self.z(self.I)];
                end
            end    
            assert(size(Zi, 1) == numel(self.I));
            assert(size(Zi, 2) == size(self.Qb, 2));

            self.Zi = Zi;
            self.q = []; % clean up
            self.z = [];
            self.a = [];

            self.sanityMLMDP();
        end

        % Given a task = reward structure for the B states,
        % compute the best combination of basis tasks
        % and the corresponding actions
        %
        function solveMLMDP(self, rb)
            assert(size(rb, 1) == numel(self.B));
            assert(size(rb, 2) == 1);
            N = numel(self.S);

            qb = exp(rb / self.lambda);
            w = pinv(self.Qb) * qb; % Eq 7 from Saxe et al (2017)

            % find desirability f'n z
            %
            z = nan(N, 1);
            zi = self.Zi * w; % specified approximately by the task and the pre-solved desirability matrix Z
            zb = qb; % specified exactly by the task
            z(self.I) = zi;
            z(self.B) = zb;
            self.z = z;

            % find optimal policy a*
            %
            a = self.policy(z);
            self.a = a;

            % Adjust the instantaneous rewards according to rb;
            % useful for the simulations
            %
            self.R(self.B) = rb;
        end

        % Sanity check that a LMDP is correct
        %
        function sanityMLMDP(self)
            % States
            %
            N = numel(self.S);
            Ni = numel(self.I);
            Nb = numel(self.B);

            % Rewards
            %
            assert(isempty(self.Zi) || size(self.Zi, 1) == Ni);
            assert(isempty(self.Zi) || size(self.Zi, 2) == Nb); % note that this is not a strict requirements for MLMDPs; it's self-imposed
            assert(size(self.Qb, 1) == Nb);
            assert(size(self.Qb, 2) == Nb); % note that this is not a strict requirements for MLMDPs; it's self-imposed
            assert(size(self.qi, 1) == Ni);
            assert(size(self.qi, 2) == 1);
            assert(isempty(self.q));
        end
    end
end
