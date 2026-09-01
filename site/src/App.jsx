import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { KeyResults } from './components/KeyResults';
import { WhyTextFails } from './components/WhyTextFails';
import { Method } from './components/Method';
import { Scorecard } from './components/Scorecard';
import { Scaling } from './components/Scaling';
import { PathA } from './components/PathA';
import { Latency } from './components/Latency';
import { TrainingMatters } from './components/TrainingMatters';
import { Ablations } from './components/Ablations';
import { Adversarial } from './components/Adversarial';
import { Pareto } from './components/Pareto';
import { CrossFamily } from './components/CrossFamily';
import { VLM } from './components/VLM';
import { CrossModal } from './components/CrossModal';
import { Mechanism } from './components/Mechanism';
import { Demo } from './components/Demo';
import { Reproducibility } from './components/Reproducibility';
import { Footer } from './components/Footer';

export default function App() {
  return (
    <div className="min-h-screen bg-paper text-ink">
      <Navbar />
      <main>
        {/* §1 Introduction */}
        <Hero />
        <KeyResults />
        {/* §2 Why text memory fails */}
        <WhyTextFails />
        {/* §3 Method */}
        <Method />
        {/* §5 Empirical Results (5.1–5.10) */}
        <Scorecard />        {/* 5.1 */}
        <Scaling />          {/* 5.2 */}
        <PathA />            {/* 5.3 */}
        <Latency />          {/* 5.4 */}
        <TrainingMatters />  {/* 5.5 */}
        <Ablations />        {/* 5.6 */}
        <Adversarial />      {/* 5.7 */}
        <Pareto />           {/* 5.7 (continued) */}
        <CrossFamily />      {/* 5.8 */}
        <VLM />              {/* 5.9 */}
        <CrossModal />       {/* 5.10 */}
        {/* Mechanism / Demo / Reproducibility */}
        <Mechanism />
        <Demo />
        <Reproducibility />
      </main>
      <Footer />
    </div>
  );
}
