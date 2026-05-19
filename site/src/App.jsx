import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { KeyResults } from './components/KeyResults';
import { Method } from './components/Method';
import { Scorecard } from './components/Scorecard';
import { Scaling } from './components/Scaling';
import { Adversarial } from './components/Adversarial';
import { Pareto } from './components/Pareto';
import { CrossFamily } from './components/CrossFamily';
import { VLM } from './components/VLM';
import { Latency } from './components/Latency';
import { Mechanism } from './components/Mechanism';
import { Demo } from './components/Demo';
import { Reproducibility } from './components/Reproducibility';
import { Footer } from './components/Footer';

export default function App() {
  return (
    <div className="min-h-screen bg-paper text-ink">
      <Navbar />
      <main>
        <Hero />
        <KeyResults />
        <Method />
        <Scorecard />
        <Scaling />
        <Adversarial />
        <Pareto />
        <CrossFamily />
        <VLM />
        <Latency />
        <Mechanism />
        <Demo />
        <Reproducibility />
      </main>
      <Footer />
    </div>
  );
}
