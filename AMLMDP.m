% Augmented Multitask MLMDP as described in Saxe et al (2017)
%
classdef AMLMDP < MLMDP

    properties (Constant = true)
        R_St_goal = 0; % reward for goal subtask state 
        R_St_nongoal = -Inf; % reward the other subtask states

        P_I_to_St = LMDP.P_I_to_B; % (normalized) move from I state to corresponding St state

    end

    properties (Access = public)
        St = []; % subtask states

        I2St = []; % = corresponding St state for given I state s, or 0 if none
        St2I = []; % = corresponding I state for given St state s
    end

    methods 

        function self = AMLMDP(map, subtask_inds)
            self = self@MLMDP(map);
            N = numel(self.S);
            Nb = numel(self.B);

            assert(isempty(intersect(map(subtask_inds), self.absorbing_symbols))); % subtask states must be distinct from boundary/absorbing states; by our design -- otherwise they conflict in I2B vs. I2St, as well as P_I_to_B vs. P_I_to_St (b/c we want those to be normalized probs)
            assert(size(subtask_inds, 1) == 1); % must be a row vector, just like S and St

            % Augment state space S with St states
            % Same for boundary states B
            %
            I_with_St = subtask_inds; % subtask squares = internal states with corresponding subtask states
            Nt = numel(I_with_St);
            St = N + 1 : N + Nt;
            S_augm = [self.S, St];
            B_augm = [self.B, St];
            N_augm = N + Nt;
            Nb_augm = Nb + Nt;

            % Mappings between I states an St states
            %
            I2St = zeros(N_augm, 1);
            St2I = zeros(N_augm, 1);
            I2St(I_with_St) = St; % mapping from I states to corresponding St states
            St2I(I2St(I2St > 0)) = I_with_St; % mapping from St states to corresponding I states
            self.I2St = I2St;
            self.St2I = St2I;
            I2B_augm = zeros(N_augm, 1);
            B2I_augm = zeros(N_augm, 1);
            I2B_augm(self.S) = self.I2B;
            I2B_augm = I2B_augm + I2St;
            B2I_augm(self.S) = self.B2I;
            B2I_augm = B2I_augm + St2I;
           
            % To be consistent
            %
            R_augm = zeros(N_augm, 1);
            R_augm(self.S) = self.R;
            R_augm(St) = self.R_St_nongoal;

            % Create Qt
            %
            Rt = ones(Nt, Nt) * self.R_St_nongoal;
            Rt(logical(eye(Nt))) = self.R_St_goal;
            Qt = exp(Rt / self.lambda);

            % Created augmented Qb = [Qb 0; 0 Qt]
            %
            Qb_augm = zeros(Nb_augm, Nb_augm);
            Qb_augm(1:Nb, 1:Nb) = self.Qb;
            Qb_augm(Nb+1:Nb+Nt, Nb+1:Nb+Nt) = Qt;

            % Augment P 
            P_augm = zeros(N_augm, N_augm);
            P_augm(self.S, self.S) = self.P;
            P_augm(:, I_with_St) = P_augm(:, I_with_St) * (1 - self.P_I_to_St); % since P_I_to_St is normalized, we need to make 'room' for it
            I_to_St = sub2ind(size(P_augm), St, I_with_St); 
            P_augm(I_to_St) = self.P_I_to_St;
            assert(sum(abs(sum(P_augm, 1) - 1) < 1e-8 | abs(sum(P_augm, 1)) < 1e-8) == N_augm);

            % Change object
            % we do this in the end b/c we need the old object as we're creating the augmented data structures
            %
            self.St = St;
            self.S = S_augm;
            self.B = B_augm;
            self.Qb = Qb_augm;
            self.P = P_augm;
            self.I2B = I2B_augm;
            self.B2I = B2I_augm;
            self.R = R_augm;
        end
    end
end
